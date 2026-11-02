/*
 * Copyright © 2023 Bas Nieuwenhuizen
 * Copyright © 2024 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "util/macros.h"
#include "util/set.h"
#include "util/u_dynarray.h"
#include "glsl_types.h"
#include "nak_private.h"
#include "nir_builder.h"

static enum nak_cmat_type
get_nak_cmat_type_for_muladd(struct glsl_cmat_description a_desc,
                             struct glsl_cmat_description b_desc,
                             struct glsl_cmat_description c_desc)
{
   unsigned m = a_desc.rows;
   unsigned k = b_desc.rows;
   unsigned n = c_desc.cols;

   bool a_is_int8 = a_desc.element_type == GLSL_TYPE_INT8 ||
                    a_desc.element_type == GLSL_TYPE_UINT8;
   bool b_is_int8 = b_desc.element_type == GLSL_TYPE_INT8 ||
                    b_desc.element_type == GLSL_TYPE_UINT8;
   bool c_is_int32 = c_desc.element_type == GLSL_TYPE_INT ||
                     c_desc.element_type == GLSL_TYPE_UINT;

   if (m ==  8 && a_is_int8 &&
       n ==  8 && b_is_int8 &&
       k == 16 && c_is_int32)
      return NAK_CMAT_TYPE_M8N8K16_INT;

   if (m == 16 && a_is_int8 &&
       n ==  8 && b_is_int8 &&
       k == 16 && c_is_int32)
      return NAK_CMAT_TYPE_M16N8K16_INT;

   if (m == 16 && a_is_int8 &&
       n ==  8 && b_is_int8 &&
       k == 32 && c_is_int32)
      return NAK_CMAT_TYPE_M16N8K32_INT;

   if (m == 16 && a_is_int8 &&
       n == 16 && b_is_int8 &&
       k == 32 && c_is_int32)
      return NAK_CMAT_TYPE_M16N16K32_INT_SW;

   if (m == 16 && a_desc.element_type == GLSL_TYPE_FLOAT16 &&
       n ==  8 && b_desc.element_type == GLSL_TYPE_FLOAT16 &&
       k ==  8 && glsl_base_type_is_float(c_desc.element_type))
      return NAK_CMAT_TYPE_M16N8K8_FLOAT;

   if (m == 16 && a_desc.element_type == GLSL_TYPE_FLOAT16 &&
       n ==  8 && b_desc.element_type == GLSL_TYPE_FLOAT16 &&
       k == 16 && glsl_base_type_is_float(c_desc.element_type))
       return NAK_CMAT_TYPE_M16N8K16_FLOAT;

   if (m == 16 && a_desc.element_type == GLSL_TYPE_FLOAT16 &&
       n == 16 && b_desc.element_type == GLSL_TYPE_FLOAT16 &&
       k == 16 && glsl_base_type_is_float(c_desc.element_type))
      return NAK_CMAT_TYPE_M16N16K16_FLOAT_SW;

   UNREACHABLE("Unable to determine matrix muladd layout!");
}

enum nak_matrix_type_layout {
   NAK_MAT_16x32_INT8,
   NAK_MAT_16x16,
};

static enum nak_matrix_type_layout
determine_matrix_type(struct glsl_cmat_description desc)
{
   bool is_int8 = desc.element_type == GLSL_TYPE_INT8 ||
                  desc.element_type == GLSL_TYPE_UINT8;
   bool is_int8_a = is_int8 && desc.use == GLSL_CMAT_USE_A;
   bool is_int8_b = is_int8 && desc.use == GLSL_CMAT_USE_B;
   ASSERTED bool is_int32 = desc.element_type == GLSL_TYPE_INT ||
                            desc.element_type == GLSL_TYPE_UINT;
   ASSERTED bool is_float16 = desc.element_type == GLSL_TYPE_FLOAT16;
   ASSERTED bool is_float32 = desc.element_type == GLSL_TYPE_FLOAT;
   ASSERTED bool use_accum = desc.use == GLSL_CMAT_USE_ACCUMULATOR;

   /* This format doesn't exist on any hardware we are aware of so far and is
    * part of lowering
    */
   if (desc.rows == 32 && desc.cols == 16 && is_int8_b)
      return NAK_MAT_16x32_INT8;

   /* Even though this condition might be correct, we assert on all the
    * combination we actually verified on hardware.
    */
   if (is_int8_a || is_int8_b) {
      assert(
         (desc.rows ==  8 && desc.cols == 16 && is_int8_a) ||
         (desc.rows == 16 && desc.cols ==  8 && is_int8_b) ||
         (desc.rows == 16 && desc.cols == 16 && is_int8_a) ||
         (desc.rows == 16 && desc.cols == 32 && is_int8_a) ||
         (desc.rows == 32 && desc.cols ==  8 && is_int8_b)
      );
      return NAK_MAT_16x32_INT8;
   } else {
      assert(
         (desc.rows ==  8 && desc.cols ==  8 && is_float16 && !use_accum) ||
         (desc.rows == 16 && desc.cols ==  8 && is_float16              ) ||
         (desc.rows == 16 && desc.cols ==  8 && is_float32              ) ||
         (desc.rows == 16 && desc.cols == 16 && is_float16              ) ||
         (desc.rows == 16 && desc.cols == 16 && is_float32              ) ||
         (desc.rows ==  8 && desc.cols ==  8 && is_int32                ) ||
         (desc.rows == 16 && desc.cols ==  8 && is_int32                ) ||
         (desc.rows == 16 && desc.cols == 16 && is_int32                )
      );
      return NAK_MAT_16x16;
   }
}

static unsigned
get_cmat_size(struct glsl_cmat_description matrix_desc)
{
   return matrix_desc.cols * matrix_desc.rows;
}

static unsigned
get_cmat_length(struct glsl_cmat_description matrix_desc)
{
   return get_cmat_size(matrix_desc) / NAK_SUBGROUP_SIZE;
}

static nir_def *
load_cmat_deref(nir_builder *b, nir_deref_instr *src)
{
   struct glsl_cmat_description matrix_desc =
      *glsl_get_cmat_description(src->type);

   return nir_build_load_deref(
      b, get_cmat_length(matrix_desc),
      glsl_base_type_bit_size(matrix_desc.element_type), &src->def, 0);
}

static ALWAYS_INLINE nir_def *
load_cmat_src(nir_builder *b, nir_src src)
{
   return load_cmat_deref(b, nir_src_as_deref(src));
}

static ALWAYS_INLINE struct glsl_cmat_description
cmat_src_desc(nir_src src)
{
   nir_deref_instr *deref = nir_src_as_deref(src);
   return *glsl_get_cmat_description(deref->type);
}

static void
store_cmat_deref(nir_builder *b, nir_deref_instr *dst, nir_def *val)
{
   ASSERTED struct glsl_cmat_description matrix_desc =
      *glsl_get_cmat_description(dst->type);

   assert(val->bit_size == glsl_base_type_bit_size(matrix_desc.element_type));
   assert(val->num_components == get_cmat_length(matrix_desc));

   nir_store_deref(b, dst, val, ~0);
}

static ALWAYS_INLINE void
store_cmat_src(nir_builder *b, nir_src dst_src, nir_def *val)
{
   store_cmat_deref(b, nir_src_as_deref(dst_src), val);
}

static const struct glsl_type *
remap_matrix_type(struct hash_table *mapping, const struct glsl_type *orig)
{
   struct hash_entry *entry = _mesa_hash_table_search(mapping, orig);

   if (entry)
      return entry->data;

   const struct glsl_type *new_type = orig;

   if (glsl_type_is_cmat(orig)) {
      struct glsl_cmat_description matrix_desc =
         *glsl_get_cmat_description(orig);

      new_type = glsl_vector_type(matrix_desc.element_type,
                                  get_cmat_length(matrix_desc));
   } else if (glsl_type_is_array(orig)) {
      const struct glsl_type *elem_type = glsl_get_array_element(orig);
      const struct glsl_type *new_elem_type =
         remap_matrix_type(mapping, elem_type);

      if (elem_type != new_elem_type) {
         new_type = glsl_array_type(new_elem_type, glsl_get_length(orig),
                                    glsl_get_explicit_stride(orig));
      }
   } else if (glsl_type_is_struct(orig)) {
      unsigned i;
      for (i = 0; i < orig->length; i++) {
         const struct glsl_type *field_type = glsl_get_struct_field(orig, i);
         const struct glsl_type *new_field_type =
            remap_matrix_type(mapping, field_type);

         if (field_type != new_field_type) {
            break;
         }
      }

      /* If we found a cmat, remap the structure type */
      if (i < orig->length) {
         struct glsl_struct_field *fields =
            malloc(sizeof(struct glsl_struct_field) * orig->length);

         /* Copy everything that didn't change */
         memcpy(fields, orig->fields.structure,
                sizeof(struct glsl_struct_field) * i);

         /* Remap the rest */
         for (; i < orig->length; i++) {
            fields[i] = *glsl_get_struct_field_data(orig, i);
            fields[i].type = remap_matrix_type(mapping, fields[i].type);
         }

         new_type =
            glsl_struct_type(fields, orig->length, glsl_get_type_name(orig),
                             glsl_struct_type_is_packed(orig));

         free(fields);
      }
   }

   _mesa_hash_table_insert(mapping, orig, (void *)new_type);
   return new_type;
}

static bool
uses_movm_for_bit_size(unsigned bit_size)
{
   return bit_size == 16;
}

/**
 * Returns true when before stores or after loads the loaded matrix has to be transposed
 */
static bool
transpose_on_load_store(struct glsl_cmat_description desc,
                        enum glsl_matrix_layout layout)
{
   return
      uses_movm_for_bit_size(glsl_base_type_get_bit_size(desc.element_type)) &&
      ((desc.use == GLSL_CMAT_USE_B && layout != GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) ||
       (desc.use != GLSL_CMAT_USE_B && layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR));
}

static nir_def *
transpose_matrix(nir_builder *b, nir_def *value)
{
   unsigned vec_size = value->num_components;
   unsigned bit_size = value->bit_size;

   switch (bit_size) {
   case 32: {
      assert(vec_size == 2);

      nir_def *raw = nir_unpack_64_4x16(b, nir_pack_64_2x32(b, value));
      nir_def *lo = nir_vec2(b,
         nir_channel(b, raw, 0),
         nir_channel(b, raw, 2)
      );
      nir_def *hi = nir_vec2(b,
         nir_channel(b, raw, 1),
         nir_channel(b, raw, 3)
      );

      lo = nir_cmat_mov_transpose_nv(b, lo);
      hi = nir_cmat_mov_transpose_nv(b, hi);

      value = nir_vec2(b,
         nir_pack_32_2x16(b, nir_vec2(b, nir_channel(b, lo, 0), nir_channel(b, hi, 0))),
         nir_pack_32_2x16(b, nir_vec2(b, nir_channel(b, lo, 1), nir_channel(b, hi, 1)))
      );
      break;
   }
   case 16:
      assert(vec_size == 2);
      value = nir_cmat_mov_transpose_nv(b, value);
      break;
   default:
      assert(!"unsupported bit_size for transpose");
      break;
   }

   return value;
}

/**
 * Computes the index in a linear matrix buffer a thread needs to load from in
 * order to execute an MMA on the Matrix.
 *
 * This is a generalized formula based on the Matrix layout descriptions from
 * the CUDA PTX instruction set documentation:
 * https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#matrix-multiply-accumulate-operation-using-mma-instruction
 *
 * \param group_size Size of the value groups the layout tiles around.
 */
static void
compute_mat(struct nir_builder *b, nir_def *lane_id,
            unsigned idx, nir_def **col, nir_def **row,
            bool alternate_tiling_order,
            unsigned group_size)
{
   assert(idx < 4 * group_size);

   nir_def *quad_id = nir_ushr_imm(b, lane_id, 2);
   nir_def *thread_id_in_quad = nir_iand_imm(b, lane_id, 0x3);

   unsigned row_bound = (alternate_tiling_order ? 2 : 1) * group_size;
   unsigned col_bound = (alternate_tiling_order ? 1 : 2) * group_size;

   *row = quad_id;
   if (idx & row_bound)
      *row = nir_iadd_imm(b, *row, 8);

   *col = nir_iadd_imm(b, nir_imul_imm(b, thread_id_in_quad, group_size),
                          idx & (group_size - 1));
   if (idx & col_bound)
      *col = nir_iadd_imm(b, *col, group_size * 4);
}

static void
compute_mat_16x32_int8(struct nir_builder *b, nir_def *lane_id,
                       unsigned idx, nir_def **col, nir_def **row,
                       bool alternate_tiling_order)
{
   compute_mat(b, lane_id, idx, col, row, alternate_tiling_order, 4);
}

static void
compute_mat_16x16(struct nir_builder *b, nir_def *lane_id,
                       unsigned idx, nir_def **col, nir_def **row,
                       bool alternate_tiling_order)
{
   compute_mat(b, lane_id, idx, col, row, alternate_tiling_order, 2);
}

static void
compute_matrix_offsets(struct nir_builder *b, struct glsl_cmat_description desc,
                       enum glsl_matrix_layout layout, nir_def *lane_id,
                       unsigned idx, nir_def **col_offset, nir_def **row_offset)
{
   enum nak_matrix_type_layout cmat_type = determine_matrix_type(desc);
   unsigned bit_size = glsl_base_type_bit_size(desc.element_type);
   bool uses_movm = uses_movm_for_bit_size(bit_size);
   bool alternate_tiling_order =
      (uses_movm && layout != GLSL_MATRIX_LAYOUT_ROW_MAJOR) ||
      (!uses_movm && desc.use == GLSL_CMAT_USE_B);

   switch (cmat_type) {
   case NAK_MAT_16x32_INT8:
      compute_mat_16x32_int8(b, lane_id, idx, col_offset, row_offset, alternate_tiling_order);
      break;

   case NAK_MAT_16x16:
      compute_mat_16x16(b, lane_id, idx, col_offset, row_offset, alternate_tiling_order);
      break;
   }

   /* The layout calculation code relies on col and row being swapped for B
    * row-major and non B col-major matrices.
    */
   if (!uses_movm) {
      if ((desc.use == GLSL_CMAT_USE_B && layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR) ||
          (desc.use != GLSL_CMAT_USE_B && layout != GLSL_MATRIX_LAYOUT_ROW_MAJOR)) {
         nir_def *tmp = *col_offset;
         *col_offset = *row_offset;
         *row_offset = tmp;
      }
   }
}

/* Returns the hw native Matrix muladd operation */
static enum nak_cmat_type
get_hw_nak_cmat_type(enum nak_cmat_type cmat_type, uint8_t sm)
{
   switch (cmat_type) {
   case NAK_CMAT_TYPE_M8N8K16_INT:
      return NAK_CMAT_TYPE_M8N8K16_INT;
   case NAK_CMAT_TYPE_M16N8K16_INT:
      return sm >= 80 ? NAK_CMAT_TYPE_M16N8K16_INT
                      : NAK_CMAT_TYPE_M8N8K16_INT; /* no lowering code yet */
   case NAK_CMAT_TYPE_M16N8K32_INT:
   case NAK_CMAT_TYPE_M16N16K32_INT_SW:
      /* On Turing we only have 8x8x16 */
      return sm >= 80 ? NAK_CMAT_TYPE_M16N8K32_INT
                      : NAK_CMAT_TYPE_M8N8K16_INT;
   case NAK_CMAT_TYPE_M16N8K8_FLOAT:
      return NAK_CMAT_TYPE_M16N8K8_FLOAT;
   case NAK_CMAT_TYPE_M16N8K16_FLOAT:
   case NAK_CMAT_TYPE_M16N16K16_FLOAT_SW:
      return NAK_CMAT_TYPE_M16N8K16_FLOAT;
   default:
      UNREACHABLE("Unknown Matrix muladd type.");
   }
}

static nir_def *
lower_cmat_muladd(nir_builder *b, nir_intrinsic_instr *intr, nir_def *cmat_a,
                  nir_def *cmat_b, nir_def *cmat_c,
                  struct glsl_cmat_description a_desc,
                  struct glsl_cmat_description b_desc,
                  struct glsl_cmat_description c_desc,
                  struct glsl_cmat_description d_desc, uint8_t sm)
{
   unsigned dst_length = get_cmat_length(d_desc);

   enum nak_cmat_type cmat_type =
      get_nak_cmat_type_for_muladd(a_desc, b_desc, c_desc);
   enum nak_cmat_type hw_cmat_type = get_hw_nak_cmat_type(cmat_type, sm);

   nir_cmat_signed cmat_signed = nir_intrinsic_cmat_signed_mask(intr);
   bool a_signed = cmat_signed & NIR_CMAT_A_SIGNED;
   bool b_signed = cmat_signed & NIR_CMAT_B_SIGNED;

   const struct nak_nir_cmat_mul_add_flags flags = {
      .cmat_type = hw_cmat_type,
      .a_type = glsl_apply_signedness_to_base_type(a_desc.element_type, a_signed),
      .b_type = glsl_apply_signedness_to_base_type(b_desc.element_type, b_signed),
      .sat = nir_intrinsic_saturate(intr),
   };

   /* Simple case: we can execute the MMA in one instruction */
   if (cmat_type == hw_cmat_type) {
      return nir_cmat_muladd_nv(b, dst_length, cmat_a, cmat_b, cmat_c,
                                .flags = NAK_AS_U32(flags));
   }

   unsigned a_length = get_cmat_length(a_desc);
   unsigned b_length = get_cmat_length(b_desc);
   unsigned c_length = get_cmat_length(c_desc);

   nir_def *a_comps[NIR_MAX_VEC_COMPONENTS];
   nir_def *b_comps[NIR_MAX_VEC_COMPONENTS];
   nir_def *c_comps[NIR_MAX_VEC_COMPONENTS];
   nir_def *d_comps[NIR_MAX_VEC_COMPONENTS];

   for (unsigned i = 0; i < a_length; i++)
      a_comps[i] = nir_channel(b, cmat_a, i);

   for (unsigned i = 0; i < b_length; i++)
      b_comps[i] = nir_channel(b, cmat_b, i);

   for (unsigned i = 0; i < c_length; i++)
      c_comps[i] = nir_channel(b, cmat_c, i);

   if (hw_cmat_type == NAK_CMAT_TYPE_M8N8K16_INT &&
         (cmat_type == NAK_CMAT_TYPE_M16N8K32_INT ||
          cmat_type == NAK_CMAT_TYPE_M16N16K32_INT_SW)) {
      const unsigned a_hw_length = 4;
      const unsigned b_hw_length = 4;
      const unsigned c_hw_length = 2;
      const unsigned d_hw_length = 2;

      for (unsigned i = 0; i < dst_length / d_hw_length; i++) {
         unsigned cmat_a_lo_offset = (i % 2) * a_hw_length;
         unsigned cmat_a_hi_offset = cmat_a_lo_offset + 8;

         unsigned cmat_b_lo_offset = (i / 2) * b_hw_length;
         if (cmat_type == NAK_CMAT_TYPE_M16N16K32_INT_SW)
            cmat_b_lo_offset *= 2;
         unsigned cmat_b_hi_offset = cmat_b_lo_offset + 4;

         unsigned cmat_c_offset = i * c_hw_length;

         nir_def *cmat_a_lo = nir_vec(b, &a_comps[cmat_a_lo_offset], a_hw_length);
         nir_def *cmat_a_hi = nir_vec(b, &a_comps[cmat_a_hi_offset], a_hw_length);
         nir_def *cmat_b_lo = nir_vec(b, &b_comps[cmat_b_lo_offset], b_hw_length);
         nir_def *cmat_b_hi = nir_vec(b, &b_comps[cmat_b_hi_offset], b_hw_length);
         nir_def *c_part = nir_vec(b, &c_comps[cmat_c_offset], c_hw_length);

         nir_def *new_c = nir_cmat_muladd_nv(b, d_hw_length, cmat_a_lo,
                                             cmat_b_lo, c_part,
                                             .flags = NAK_AS_U32(flags));
         nir_def *tmp_d = nir_cmat_muladd_nv(b, d_hw_length, cmat_a_hi,
                                             cmat_b_hi, new_c,
                                             .flags = NAK_AS_U32(flags));

         for (unsigned c = 0; c < d_hw_length; c++)
            d_comps[i * d_hw_length + c] = nir_channel(b, tmp_d, c);
      }
   } else if ((cmat_type == NAK_CMAT_TYPE_M16N16K32_INT_SW &&
               hw_cmat_type == NAK_CMAT_TYPE_M16N8K32_INT) ||
              (cmat_type == NAK_CMAT_TYPE_M16N16K16_FLOAT_SW &&
               hw_cmat_type == NAK_CMAT_TYPE_M16N8K16_FLOAT))  {
      nir_def *cmat_b_lo = nir_vec(b,  b_comps,               b_length / 2);
      nir_def *cmat_b_hi = nir_vec(b, &b_comps[b_length / 2], b_length / 2);

      nir_def *cmat_c_lo = nir_vec(b,  c_comps,               c_length / 2);
      nir_def *cmat_c_hi = nir_vec(b, &c_comps[c_length / 2], c_length / 2);

      nir_def *cmat_d_lo = nir_cmat_muladd_nv(b, dst_length / 2, cmat_a,
                                              cmat_b_lo, cmat_c_lo,
                                              .flags = NAK_AS_U32(flags));
      nir_def *cmat_d_hi = nir_cmat_muladd_nv(b, dst_length / 2, cmat_a,
                                              cmat_b_hi, cmat_c_hi,
                                              .flags = NAK_AS_U32(flags));

      for (unsigned i = 0; i < dst_length / 2; i++) {
         d_comps[i]                  = nir_channel(b, cmat_d_lo, i);
         d_comps[i + dst_length / 2] = nir_channel(b, cmat_d_hi, i);
      }
   } else {
      assert(0 && "lowering not implemented");
   }

   return nir_vec(b, d_comps, dst_length);
}

static nir_def *
lower_cmat_convert(nir_builder *b, nir_intrinsic_instr *intr, nir_def *cmat,
                   struct glsl_cmat_description a_desc,
                   struct glsl_cmat_description d_desc)
{
   nir_cmat_signed cmat_signed_mask = nir_intrinsic_cmat_signed_mask(intr);

   enum glsl_base_type src_type = glsl_apply_signedness_to_base_type(
      a_desc.element_type, cmat_signed_mask & NIR_CMAT_A_SIGNED);
   enum glsl_base_type dst_type = glsl_apply_signedness_to_base_type(
      d_desc.element_type, cmat_signed_mask & NIR_CMAT_RESULT_SIGNED);

   /* We want to shuffle the smaller values for better packing. */
   bool conv_narrows =
      glsl_base_type_bit_size(src_type) > glsl_base_type_bit_size(dst_type);
   nir_op op =
      nir_type_conversion_op(nir_get_nir_type_for_glsl_base_type(src_type),
                             nir_get_nir_type_for_glsl_base_type(dst_type),
                             nir_rounding_mode_undef);

   /* If the result type is smaller, we convert before shuffling. */
   if (conv_narrows)
      cmat = nir_build_alu1(b, op, cmat);

   enum nak_matrix_type_layout a_layout = determine_matrix_type(a_desc);
   enum nak_matrix_type_layout d_layout = determine_matrix_type(d_desc);

   /* Matrix layout conversion code. For some conversions we also need
    * to fix the layout, so we shuffle values around to achieve that.
    */
   if (a_layout != d_layout) {
      nir_def *lane_id = nir_load_subgroup_invocation(b);
      unsigned mask    = a_layout == NAK_MAT_16x16 ? 0x1 : 0x2;
      unsigned compare = a_layout == NAK_MAT_16x16 ? 0x2 : 0x1;

      nir_def *adj;
      if (a_layout == NAK_MAT_16x16) {
         adj = nir_ishl_imm(b, nir_iand_imm(b, lane_id, mask), 1);
      } else {
         adj = nir_ushr_imm(b, nir_iand_imm(b, lane_id, mask), 1);
      }

      /* lane_id & 0x1c + (lane_id & mask << 1) */
      /* lane_id & 0x1c + (lane_id & mask >> 1) */
      nir_def *lane0 = nir_iadd(b, nir_iand_imm(b, lane_id, 0x1c), adj);
      /* lane_id & 0x1c + (lane_id & mask << 1) + mask */
      /* lane_id & 0x1c + (lane_id & mask >> 1) + mask */
      nir_def *lane1 = nir_iadd_imm(b, lane0, mask);
      nir_def *cond = nir_ieq_imm(b, nir_iand_imm(b, lane_id, compare), 0);

      if (cmat->num_components == 4) {
         nir_def *xy = nir_channels(b, cmat, 0x3);
         nir_def *zw = nir_channels(b, cmat, 0xc);

         nir_def *xy0 = nir_shuffle(b, xy, lane0);
         nir_def *zw0 = nir_shuffle(b, xy, lane1);
         nir_def *xy1 = nir_shuffle(b, zw, lane0);
         nir_def *zw1 = nir_shuffle(b, zw, lane1);

         xy = nir_bcsel(b, cond, xy0, xy1);
         zw = nir_bcsel(b, cond, zw0, zw1);

         cmat = nir_vec4(b,
            nir_channel(b, xy, 0),
            nir_channel(b, xy, 1),
            nir_channel(b, zw, 0),
            nir_channel(b, zw, 1)
         );
      } else if (cmat->num_components == 8 && a_layout == NAK_MAT_16x16) {
         nir_def *abcd = nir_channels(b, cmat, 0x0f);
         nir_def *efgh = nir_channels(b, cmat, 0xf0);

         nir_def *abef0 = nir_shuffle(b, abcd, lane0);
         nir_def *cdgh0 = nir_shuffle(b, abcd, lane1);
         nir_def *abef1 = nir_shuffle(b, efgh, lane0);
         nir_def *cdgh1 = nir_shuffle(b, efgh, lane1);

         nir_def *abef = nir_bcsel(b, cond, abef0, abef1);
         nir_def *cdgh = nir_bcsel(b, cond, cdgh0, cdgh1);

         cmat = nir_vec8(b,
            nir_channel(b, abef, 0),
            nir_channel(b, abef, 1),
            nir_channel(b, cdgh, 0),
            nir_channel(b, cdgh, 1),
            nir_channel(b, abef, 2),
            nir_channel(b, abef, 3),
            nir_channel(b, cdgh, 2),
            nir_channel(b, cdgh, 3)
         );
      } else if (cmat->num_components == 8 && a_layout == NAK_MAT_16x32_INT8) {
         nir_def *abef = nir_channels(b, cmat, 0x33);
         nir_def *cdgh = nir_channels(b, cmat, 0xcc);

         nir_def *abcd0 = nir_shuffle(b, abef, lane0);
         nir_def *efgh0 = nir_shuffle(b, abef, lane1);
         nir_def *abcd1 = nir_shuffle(b, cdgh, lane0);
         nir_def *efgh1 = nir_shuffle(b, cdgh, lane1);

         nir_def *abcd = nir_bcsel(b, cond, abcd0, abcd1);
         nir_def *efgh = nir_bcsel(b, cond, efgh0, efgh1);

         cmat = nir_vec8(b,
            nir_channel(b, abcd, 0),
            nir_channel(b, abcd, 1),
            nir_channel(b, abcd, 2),
            nir_channel(b, abcd, 3),
            nir_channel(b, efgh, 0),
            nir_channel(b, efgh, 1),
            nir_channel(b, efgh, 2),
            nir_channel(b, efgh, 3)
         );
      } else {
         UNREACHABLE("unsupported component counts for Matrix layout conversion");
      }
   }

   /* If the result type is not smaller, we convert after shuffling */
   if (!conv_narrows)
      cmat = nir_build_alu1(b, op, cmat);

   return cmat;
}

static struct nir_def*
try_lower_cmat_load_to_ldsm(nir_builder *b, nir_intrinsic_instr *intr)
{
   assert(intr->intrinsic == nir_intrinsic_cmat_load);

   enum glsl_matrix_layout layout = nir_intrinsic_matrix_layout(intr);

   const struct glsl_cmat_description desc = cmat_src_desc(intr->src[0]);
   const unsigned length = get_cmat_length(desc);
   nir_deref_instr *deref = nir_def_as_deref(intr->src[1].ssa);
   const unsigned ptr_bit_size = glsl_get_bit_size(deref->type);
   const unsigned vec = glsl_get_vector_elements(deref->type);
   nir_src stride = intr->src[2];

   /* Even though LDSM operates on 16 bit types, the int8 matrix layout is
    * compatible so that we can use LDSM on it as well. But we can't use it on
    * the 32 bit types, because that actually uses a different data layout on a
    * byte level.
    */
   const unsigned bit_size = glsl_base_type_bit_size(desc.element_type);
   if (!nir_src_is_const(stride)
       || !nir_deref_mode_is(deref, nir_var_mem_shared)
       || bit_size > 16)
       return NULL;

   /* The stride is in elements of the pointed to type, not necessarily the
    * type of the referenced matrix
    */
   unsigned stride_bytes = nir_src_as_uint(stride) * vec * ptr_bit_size / 8;
   if (stride_bytes % 16 != 0)
      return NULL;

   /* check implicit base ptr alignment */
   if ((layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR    && desc.cols * bit_size < 128) ||
       (layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR && desc.rows * bit_size < 128))
         return NULL;

   /* LDSM loads n 8x8 16 bit matrices */
   unsigned mat_size_bits = desc.rows * desc.cols * bit_size;
   unsigned ldsm_count = mat_size_bits / (8 * 8 * 16);

   /* TODO: split bigger ones into multiple LDSM calls */
   if (ldsm_count > 4 || ldsm_count == 0)
      return NULL;

   if ((desc.use != GLSL_CMAT_USE_B && layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) ||
       (desc.use == GLSL_CMAT_USE_B && layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR)) {
      /* A row-major 16x16 B is four transposed 8x8 loads whose rows the
       * row-major offsets below already give in the order the column-major
       * case loads them: k 0-7 and 8-15 of columns 0-7, then of 8-15.
       * Column-major A would need another order.  Quite the pain, might
       * not be worth it.
       */
      if (ldsm_count >= 4 && desc.use != GLSL_CMAT_USE_B)
         return NULL;

      /* We'd need to split the rows leading to unaligned loads */
      if (ldsm_count >= 2 && (desc.rows / 2) * bit_size < 128)
         return NULL;
   }

   /* Account for differences in tiling depending on the layout */
   nir_def *offset;
   nir_def *lane_id = nir_load_subgroup_invocation(b);
   if (ldsm_count == 4 && layout != GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) {
      nir_def *lower = nir_iand(b, lane_id, nir_imm_int(b, 0x0f));
      nir_def *upper = nir_iand(b, lane_id, nir_imm_int(b, 0x10));

      offset = nir_imul_imm(b, lower, stride_bytes);
      offset = nir_iadd(b, offset, upper);
   } else if (ldsm_count >= 2 && layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) {
      nir_def *lower;
      nir_def *lower_lo = nir_iand(b, lane_id, nir_imm_int(b, 0x07));
      nir_def *upper = nir_iand(b, lane_id, nir_imm_int(b, 0x08));
      if (ldsm_count == 4) {
         nir_def *lower_hi = nir_iand(b, lane_id, nir_imm_int(b, 0x10));
         lower = nir_ior(b, lower_lo, nir_ushr_imm(b, lower_hi, 1));
      } else {
         lower = lower_lo;
      }

      offset = nir_imul_imm(b, lower, stride_bytes);
      offset = nir_iadd(b, offset, nir_ishl_imm(b, upper, 1));
   } else {
      offset = nir_imul_imm(b, lane_id, stride_bytes);
   }

   nir_def *base = intr->src[1].ssa;
   offset = nir_u2uN(b, offset, base->bit_size);
   nir_def *addr = nir_iadd(b, base, offset);
   nir_def *zero = nir_imm_zero(b, addr->num_components, addr->bit_size);

   /* flip the layout for B matrices */
   if (desc.use == GLSL_CMAT_USE_B) {
      if (layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR)
         layout = GLSL_MATRIX_LAYOUT_COLUMN_MAJOR;
      else if (layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR)
         layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR;
   }

   /* Each thread loads 32 bits per matrix */
   assert(length * bit_size == 32 * ldsm_count);
   return nir_cmat_load_shared_nv(b, length, bit_size, addr, zero,
                                     .num_matrices = ldsm_count,
                                     .matrix_layout = layout);
}

/**
 * Returns the possibly vectorization width we can use to load/store matrices
 * of the given cmat desc and layout
 */
static int load_store_get_vec_size(const struct glsl_cmat_description desc,
                                   enum glsl_matrix_layout layout)
{
   unsigned bit_size = glsl_base_type_bit_size(desc.element_type);
   bool uses_movm = uses_movm_for_bit_size(bit_size);
   bool needs_transpose =
      (desc.use != GLSL_CMAT_USE_B && layout != GLSL_MATRIX_LAYOUT_ROW_MAJOR) ||
      (desc.use == GLSL_CMAT_USE_B && layout != GLSL_MATRIX_LAYOUT_COLUMN_MAJOR);

   if (needs_transpose && !uses_movm)
      return 1;

   switch (bit_size) {
   case 16:
   case 32:
      return 2;
   case  8:
      return 4;
   default:
      return 1;
   }
}

static nir_deref_instr*
get_cmat_component_deref(nir_builder *b, nir_intrinsic_instr *intr,
                         nir_def *lane_id, unsigned idx)
{
   unsigned deref_src = intr->intrinsic == nir_intrinsic_cmat_store ? 0 : 1;
   unsigned cmat_src = intr->intrinsic == nir_intrinsic_cmat_store ? 1 : 0;

   const struct glsl_cmat_description desc = cmat_src_desc(intr->src[cmat_src]);
   nir_deref_instr *deref = nir_def_as_deref(intr->src[deref_src].ssa);
   unsigned type_size_B = glsl_base_type_bit_size(desc.element_type) / 8;

   const enum glsl_matrix_layout layout = nir_intrinsic_matrix_layout(intr);
   nir_def *stride = intr->src[2].ssa;

   nir_def *col_offset;
   nir_def *row_offset;
   compute_matrix_offsets(b, desc, layout, lane_id, idx,
                              &col_offset, &row_offset);

   row_offset = nir_imul(b, row_offset, stride);
   col_offset = nir_u2uN(b, col_offset, deref->def.bit_size);
   row_offset = nir_u2uN(b, row_offset, deref->def.bit_size);

   unsigned align_mul = 0, align_offset = 0, combined_align = 0;
   nir_get_explicit_deref_align(deref, false, &align_mul, &align_offset);

   if (align_mul)
      combined_align = nir_combined_align(align_mul, align_offset);

   /* VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986:
    * For OpCooperativeMatrixLoadKHR and OpCooperativeMatrixStoreKHR
    * instructions, the Pointer and Stride operands must be aligned to at least
    * the lesser of 16 bytes or the natural alignment of a row or column
    * (depending on ColumnMajor) of the matrix (where the natural alignment is
    * the number of columns/rows multiplied by the component size) */
   unsigned align_elems =
      layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR ? desc.rows : desc.cols;
   unsigned implicit_align = MIN2(16, align_elems * type_size_B);
   if (implicit_align > combined_align) {
      align_mul = implicit_align;
      align_offset = 0;
   }

   /* We have to ignore the incoming stride, but have to choose the type of
    * the pointer as the declared stride is in multiple of the pointer type */
   deref = nir_build_deref_cast_with_alignment(
      b, &deref->def, deref->modes,
      deref->type,
      glsl_get_vector_elements(deref->type) * glsl_get_bit_size(deref->type) / 8,
      align_mul,
      align_offset
   );
   deref = nir_build_deref_ptr_as_array(b, deref, row_offset);
   deref = nir_build_deref_cast(
      b, &deref->def, deref->modes,
      glsl_scalar_type(desc.element_type),
      type_size_B);
   return nir_build_deref_ptr_as_array(b, deref, col_offset);
}

static void
lower_cmat_load(nir_builder *b, nir_intrinsic_instr *intr)
{
   struct nir_def *ldsm = try_lower_cmat_load_to_ldsm(b, intr);
   if (ldsm) {
      store_cmat_src(b, intr->src[0], ldsm);
      return;
   }

   const struct glsl_cmat_description desc = cmat_src_desc(intr->src[0]);
   const enum glsl_matrix_layout layout = nir_intrinsic_matrix_layout(intr);
   const unsigned length = get_cmat_length(desc);

   nir_def *vars[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < length; ++i)
      vars[i] = nir_undef(b, 1, glsl_base_type_bit_size(desc.element_type));

   nir_def *lane_id = nir_load_subgroup_invocation(b);

   int vec_size = load_store_get_vec_size(desc, layout);
   for (unsigned idx = 0; idx < length; idx += vec_size) {
      nir_deref_instr *iter_deref =
         get_cmat_component_deref(b, intr, lane_id, idx);
      nir_variable_mode modes = iter_deref->modes;
      const glsl_type *vec_type = glsl_vector_type(desc.element_type, vec_size);
      iter_deref = nir_build_deref_cast_with_alignment(b,
         &iter_deref->def, modes, vec_type,
         0, vec_size * glsl_base_type_bit_size(desc.element_type) / 8, 0);

      nir_def *value = nir_load_deref(b, iter_deref);
      if (transpose_on_load_store(desc, layout))
         value = transpose_matrix(b, value);

      for (int c = 0; c < vec_size; c++)
         vars[idx + c] = nir_channel(b, value, c);
   }

   nir_def *mat = nir_vec(b, vars, length);
   store_cmat_src(b, intr->src[0], mat);
}

/* Register layout of a fragment: the PTX mma operand layouts, which is what
 * the loads above produce for every memory layout.  A and accumulators are
 * held by row, B by column.  A lane holds runs of group_size consecutive
 * elements: 4 for 8-bit A/B and 2 otherwise.
 */
static unsigned
cmat_group_size(struct glsl_cmat_description desc)
{
   return glsl_base_type_bit_size(desc.element_type) == 8 ? 4 : 2;
}

static void
cmat_coord_imm(struct glsl_cmat_description desc, unsigned lane, unsigned i,
               unsigned *row, unsigned *col)
{
   const unsigned gs = cmat_group_size(desc);
   const unsigned g = lane >> 2, t = lane & 3;

   if (desc.use == GLSL_CMAT_USE_B) {
      *row = t * gs + (i % gs) + 4 * gs * ((i / gs) & 1);
      *col = g + 8 * (i / (2 * gs));
   } else {
      *row = g + 8 * ((i / gs) & 1);
      *col = t * gs + (i % gs) + 4 * gs * (i / (2 * gs));
   }
}

static void
cmat_coord(nir_builder *b, struct glsl_cmat_description desc,
           nir_def *lane, nir_def *i, nir_def **row, nir_def **col)
{
   const unsigned gs = cmat_group_size(desc);
   const unsigned gs_log2 = util_logbase2(gs);

   nir_def *g = nir_ushr_imm(b, lane, 2);
   nir_def *t = nir_iand_imm(b, lane, 3);
   nir_def *in_group = nir_iand_imm(b, i, gs - 1);
   nir_def *half = nir_iand_imm(b, nir_ushr_imm(b, i, gs_log2), 1);
   nir_def *outer = nir_ushr_imm(b, i, gs_log2 + 1);

   nir_def *run = nir_iadd(b, nir_imul_imm(b, t, gs), in_group);
   if (desc.use == GLSL_CMAT_USE_B) {
      *row = nir_iadd(b, run, nir_imul_imm(b, half, 4 * gs));
      *col = nir_iadd(b, g, nir_imul_imm(b, outer, 8));
   } else {
      *row = nir_iadd(b, g, nir_imul_imm(b, half, 8));
      *col = nir_iadd(b, run, nir_imul_imm(b, outer, 4 * gs));
   }
}

/* Inverse of cmat_coord: the lane holding an element and its index there */
static void
cmat_lane_idx_imm(struct glsl_cmat_description desc, unsigned row, unsigned col,
                  unsigned *lane, unsigned *idx)
{
   const unsigned gs = cmat_group_size(desc);

   if (desc.use == GLSL_CMAT_USE_B) {
      *lane = (col & 7) * 4 + (row % (4 * gs)) / gs;
      *idx = row % gs + gs * ((row / (4 * gs)) & 1) + 2 * gs * (col >> 3);
   } else {
      *lane = (row & 7) * 4 + (col % (4 * gs)) / gs;
      *idx = col % gs + gs * ((row >> 3) & 1) + 2 * gs * (col / (4 * gs));
   }
}

static void
cmat_lane_idx(nir_builder *b, struct glsl_cmat_description desc,
              nir_def *row, nir_def *col, nir_def **lane, nir_def **idx)
{
   const unsigned gs = cmat_group_size(desc);
   const unsigned gs_log2 = util_logbase2(gs);

   /* Rows of A/C and columns of B are spread over the quads */
   nir_def *spread = desc.use == GLSL_CMAT_USE_B ? col : row;
   nir_def *run = desc.use == GLSL_CMAT_USE_B ? row : col;

   nir_def *in_quad = nir_ushr_imm(b, nir_iand_imm(b, run, 4 * gs - 1), gs_log2);
   *lane = nir_iadd(b, nir_imul_imm(b, nir_iand_imm(b, spread, 7), 4), in_quad);

   nir_def *spread_half = nir_iand_imm(b, nir_ushr_imm(b, spread, 3), 1);
   nir_def *run_half = nir_ushr_imm(b, run, gs_log2 + 2);
   if (desc.use == GLSL_CMAT_USE_B)
      SWAP(spread_half, run_half);
   *idx = nir_iadd(b, nir_iand_imm(b, run, gs - 1),
                   nir_iadd(b, nir_imul_imm(b, spread_half, gs),
                            nir_imul_imm(b, run_half, 2 * gs)));
}

/* Fetches the element of src at (row, col) for every lane.  rows/cols are
 * the same coordinates per lane at compile time, which lets us skip the
 * shuffle when every lane reads its own data and pick only between the
 * source components some lane actually needs.
 */
static nir_def *
cmat_gather(nir_builder *b, struct glsl_cmat_description src_desc,
            nir_def **src, const unsigned *rows, const unsigned *cols,
            nir_def *row, nir_def *col)
{
   bool same_lane = true;
   uint32_t idx_mask = 0;
   for (unsigned l = 0; l < NAK_SUBGROUP_SIZE; l++) {
      assert(rows[l] < src_desc.rows && cols[l] < src_desc.cols);
      unsigned lane, idx;
      cmat_lane_idx_imm(src_desc, rows[l], cols[l], &lane, &idx);
      same_lane &= lane == l;
      idx_mask |= BITFIELD_BIT(idx);
   }

   nir_def *lane, *idx;
   cmat_lane_idx(b, src_desc, row, col, &lane, &idx);

   nir_def *res = NULL;
   u_foreach_bit(v, idx_mask) {
      nir_def *val = same_lane ? src[v] : nir_shuffle(b, src[v], lane);
      res = res ? nir_bcsel(b, nir_ieq_imm(b, idx, v), val, res) : val;
   }
   return res;
}

static void
cmat_split(nir_builder *b, nir_def *mat, nir_def **comps)
{
   for (unsigned i = 0; i < mat->num_components; i++)
      comps[i] = nir_channel(b, mat, i);
}

/* Conversions that change the use and/or transpose: move every element to
 * the lane and component the destination layout wants it in.
 */
static nir_def *
lower_cmat_relayout(nir_builder *b, nir_intrinsic_instr *intr, nir_def *cmat,
                    struct glsl_cmat_description src_desc,
                    struct glsl_cmat_description dst_desc, bool transpose)
{
   nir_cmat_signed cmat_signed_mask = nir_intrinsic_cmat_signed_mask(intr);
   enum glsl_base_type src_type = glsl_apply_signedness_to_base_type(
      src_desc.element_type, cmat_signed_mask & NIR_CMAT_A_SIGNED);
   enum glsl_base_type dst_type = glsl_apply_signedness_to_base_type(
      dst_desc.element_type, cmat_signed_mask & NIR_CMAT_RESULT_SIGNED);
   nir_op op =
      nir_type_conversion_op(nir_get_nir_type_for_glsl_base_type(src_type),
                             nir_get_nir_type_for_glsl_base_type(dst_type),
                             nir_rounding_mode_undef);
   bool conv_narrows =
      glsl_base_type_bit_size(src_type) > glsl_base_type_bit_size(dst_type);

   assert(transpose ? (src_desc.rows == dst_desc.cols &&
                       src_desc.cols == dst_desc.rows)
                    : (src_desc.rows == dst_desc.rows &&
                       src_desc.cols == dst_desc.cols));

   /* Shuffle the smaller values */
   if (conv_narrows && op != nir_op_mov)
      cmat = nir_build_alu1(b, op, cmat);

   nir_def *src[NIR_MAX_VEC_COMPONENTS];
   cmat_split(b, cmat, src);

   nir_def *lane_id = nir_load_subgroup_invocation(b);
   const unsigned dst_length = get_cmat_length(dst_desc);
   nir_def *dst[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < dst_length; i++) {
      unsigned rows[NAK_SUBGROUP_SIZE], cols[NAK_SUBGROUP_SIZE];
      for (unsigned l = 0; l < NAK_SUBGROUP_SIZE; l++) {
         cmat_coord_imm(dst_desc, l, i, &rows[l], &cols[l]);
         if (transpose)
            SWAP(rows[l], cols[l]);
      }

      nir_def *row, *col;
      cmat_coord(b, dst_desc, lane_id, nir_imm_int(b, i), &row, &col);
      if (transpose)
         SWAP(row, col);

      dst[i] = cmat_gather(b, src_desc, src, rows, cols, row, col);
   }
   cmat = nir_vec(b, dst, dst_length);

   if (!conv_narrows && op != nir_op_mov)
      cmat = nir_build_alu1(b, op, cmat);

   return cmat;
}

static nir_def *
cmat_combine(nir_builder *b, nir_function *fn, nir_def *x, nir_def *y)
{
   nir_variable *tmp =
      nir_local_variable_create(b->impl,
                                glsl_get_bare_type(fn->params[0].type),
                                "cmat_reduce_tmp");
   nir_deref_instr *deref = nir_build_deref_var(b, tmp);
   nir_call(b, fn, &deref->def, x, y);
   return nir_load_deref(b, deref);
}

static nir_def *
cmat_combine_xor(nir_builder *b, nir_function *fn, nir_def *x,
                 unsigned first_mask, unsigned last_mask)
{
   for (unsigned mask = first_mask; mask <= last_mask; mask <<= 1)
      x = cmat_combine(b, fn, x, nir_shuffle_xor(b, x, nir_imm_int(b, mask)));
   return x;
}

/* Fills dst from a fragment in which every element of the reduced rows
 * and/or columns already holds the reduction.
 */
static void
store_reduced(nir_builder *b, nir_deref_instr *dst_deref,
              struct glsl_cmat_description src_desc, nir_def **src,
              nir_cmat_reduce reduce)
{
   const struct glsl_cmat_description dst_desc =
      *glsl_get_cmat_description(dst_deref->type);
   const unsigned dst_length = get_cmat_length(dst_desc);
   nir_def *dst[NIR_MAX_VEC_COMPONENTS];

   if (dst_desc.rows == src_desc.rows && dst_desc.cols == src_desc.cols &&
       dst_desc.use == src_desc.use) {
      memcpy(dst, src, dst_length * sizeof(*dst));
   } else {
      nir_def *lane_id = nir_load_subgroup_invocation(b);
      for (unsigned i = 0; i < dst_length; i++) {
         unsigned rows[NAK_SUBGROUP_SIZE], cols[NAK_SUBGROUP_SIZE];
         for (unsigned l = 0; l < NAK_SUBGROUP_SIZE; l++) {
            cmat_coord_imm(dst_desc, l, i, &rows[l], &cols[l]);
            if (!(reduce & NIR_CMAT_REDUCE_ROW))
               rows[l] = 0;
            if (!(reduce & NIR_CMAT_REDUCE_COLUMN))
               cols[l] = 0;
         }

         nir_def *row, *col;
         cmat_coord(b, dst_desc, lane_id, nir_imm_int(b, i), &row, &col);
         if (!(reduce & NIR_CMAT_REDUCE_ROW))
            row = nir_imm_int(b, 0);
         if (!(reduce & NIR_CMAT_REDUCE_COLUMN))
            col = nir_imm_int(b, 0);

         dst[i] = cmat_gather(b, src_desc, src, rows, cols, row, col);
      }
   }

   store_cmat_deref(b, dst_deref, nir_vec(b, dst, dst_length));
}

static void
lower_cmat_reduce(nir_builder *b, nir_cmat_call_instr *call)
{
   nir_deref_instr *src_deref = nir_src_as_deref(call->params[1]);
   const struct glsl_cmat_description desc =
      *glsl_get_cmat_description(src_deref->type);
   const nir_cmat_reduce reduce = nir_cmat_call_reduce_flags(call);
   const unsigned length = get_cmat_length(desc);
   const unsigned gs = cmat_group_size(desc);
   nir_function *fn = call->callee;

   assert(desc.use == GLSL_CMAT_USE_ACCUMULATOR);
   assert(!(reduce & NIR_CMAT_REDUCE_2X2));

   nir_def *v[NIR_MAX_VEC_COMPONENTS];
   cmat_split(b, load_cmat_deref(b, src_deref), v);

   if (reduce & NIR_CMAT_REDUCE_ROW) {
      /* A lane holds two rows; the four lanes of a quad share them */
      nir_def *red[2] = { NULL, NULL };
      for (unsigned i = 0; i < length; i++) {
         unsigned h = (i / gs) & 1;
         red[h] = red[h] ? cmat_combine(b, fn, red[h], v[i]) : v[i];
      }
      for (unsigned h = 0; h < 2; h++) {
         if (red[h])
            red[h] = cmat_combine_xor(b, fn, red[h], 1, 2);
      }
      for (unsigned i = 0; i < length; i++)
         v[i] = red[(i / gs) & 1];
   }

   if (reduce & NIR_CMAT_REDUCE_COLUMN) {
      /* Element i and i + gs are the same column eight rows apart; the
       * rows are spread over the eight quads.
       */
      for (unsigned i = 0; i < length; i++) {
         if ((i / gs) & 1)
            continue;
         nir_def *red = v[i];
         if (i + gs < length)
            red = cmat_combine(b, fn, red, v[i + gs]);
         red = cmat_combine_xor(b, fn, red, 4, 16);
         v[i] = red;
         if (i + gs < length)
            v[i + gs] = red;
      }
   }

   store_reduced(b, nir_src_as_deref(call->params[0]), desc, v, reduce);
}

static void
lower_cmat_reduce_finish(nir_builder *b, nir_cmat_call_instr *call)
{
   nir_def *src0 = load_cmat_src(b, call->params[1]);
   nir_def *src1 = load_cmat_src(b, call->params[2]);

   nir_def *v[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < src0->num_components; i++) {
      v[i] = cmat_combine(b, call->callee, nir_channel(b, src0, i),
                          nir_channel(b, src1, i));
   }

   store_cmat_src(b, call->params[0], nir_vec(b, v, src0->num_components));
}

/* dst is the 2x2 reduction of the matrix made of src0..3 (top left, top
 * right, bottom left, bottom right), each the size of dst.
 */
static void
lower_cmat_reduce_2x2(nir_builder *b, nir_cmat_call_instr *call)
{
   nir_deref_instr *dst_deref = nir_src_as_deref(call->params[0]);
   const struct glsl_cmat_description desc =
      *glsl_get_cmat_description(dst_deref->type);
   const unsigned length = get_cmat_length(desc);
   const unsigned half_rows = desc.rows / 2, half_cols = desc.cols / 2;

   nir_def *src[4][NIR_MAX_VEC_COMPONENTS];
   for (unsigned q = 0; q < 4; q++) {
      ASSERTED struct glsl_cmat_description src_desc =
         cmat_src_desc(call->params[q + 1]);
      assert(src_desc.rows == desc.rows && src_desc.cols == desc.cols &&
             src_desc.use == desc.use);
      cmat_split(b, load_cmat_src(b, call->params[q + 1]), src[q]);
   }

   nir_def *lane_id = nir_load_subgroup_invocation(b);
   nir_def *v[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < length; i++) {
      /* The quadrant only depends on the component */
      unsigned r0, c0;
      cmat_coord_imm(desc, 0, i, &r0, &c0);
      const unsigned q_row = r0 / half_rows, q_col = c0 / half_cols;

      unsigned rows[NAK_SUBGROUP_SIZE], cols[NAK_SUBGROUP_SIZE];
      for (unsigned l = 0; l < NAK_SUBGROUP_SIZE; l++) {
         cmat_coord_imm(desc, l, i, &rows[l], &cols[l]);
         assert(rows[l] / half_rows == q_row && cols[l] / half_cols == q_col);
         rows[l] = (rows[l] - q_row * half_rows) * 2;
         cols[l] = (cols[l] - q_col * half_cols) * 2;
      }

      nir_def *row, *col;
      cmat_coord(b, desc, lane_id, nir_imm_int(b, i), &row, &col);
      row = nir_ishl_imm(b, nir_iadd_imm(b, row, -(int)(q_row * half_rows)), 1);
      col = nir_ishl_imm(b, nir_iadd_imm(b, col, -(int)(q_col * half_cols)), 1);

      nir_def **s = src[q_row * 2 + q_col];
      nir_def *red = NULL;
      for (unsigned d = 0; d < 4; d++) {
         unsigned dr = d >> 1, dc = d & 1;
         unsigned rows_d[NAK_SUBGROUP_SIZE], cols_d[NAK_SUBGROUP_SIZE];
         for (unsigned l = 0; l < NAK_SUBGROUP_SIZE; l++) {
            rows_d[l] = rows[l] + dr;
            cols_d[l] = cols[l] + dc;
         }
         nir_def *val = cmat_gather(b, desc, s, rows_d, cols_d,
                                    nir_iadd_imm(b, row, dr),
                                    nir_iadd_imm(b, col, dc));
         red = red ? cmat_combine(b, call->callee, red, val) : val;
      }
      v[i] = red;
   }

   store_cmat_deref(b, dst_deref, nir_vec(b, v, length));
}

static void
lower_cmat_per_element_op(nir_builder *b, nir_cmat_call_instr *call)
{
   nir_deref_instr *dst_deref = nir_src_as_deref(call->params[0]);
   const struct glsl_cmat_description desc =
      *glsl_get_cmat_description(dst_deref->type);
   const unsigned length = get_cmat_length(desc);
   const unsigned bit_size = glsl_base_type_bit_size(desc.element_type);

   nir_def *src = load_cmat_src(b, call->params[3]);

   /* Extra matrix operands are passed element by element */
   nir_def *extra[call->num_params];
   for (unsigned p = 0; p < call->num_params; p++) {
      extra[p] = NULL;
      if (p < 4)
         continue;

      nir_deref_instr *deref = nir_src_as_deref(call->params[p]);
      if (deref && glsl_type_is_cmat(deref->type))
         extra[p] = load_cmat_deref(b, deref);
   }

   nir_variable *elem_var =
      nir_local_variable_create(b->impl, glsl_get_cmat_element(dst_deref->type),
                                "cmat_elem_tmp");
   nir_deref_instr *elem_deref = nir_build_deref_var(b, elem_var);
   nir_def *lane_id = nir_load_subgroup_invocation(b);

   nir_def *v[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < length; i++) {
      nir_def *row, *col;
      cmat_coord(b, desc, lane_id, nir_imm_int(b, i), &row, &col);

      nir_call_instr *elem_call = nir_call_instr_create(b->shader, call->callee);
      elem_call->params[0] = nir_src_for_ssa(&elem_deref->def);
      elem_call->params[1] = nir_src_for_ssa(nir_iadd(b, call->params[1].ssa, row));
      elem_call->params[2] = nir_src_for_ssa(nir_iadd(b, call->params[2].ssa, col));
      elem_call->params[3] = nir_src_for_ssa(nir_channel(b, src, i));
      for (unsigned p = 4; p < call->num_params; p++) {
         nir_def *def = extra[p] ? nir_channel(b, extra[p], i)
                                 : call->params[p].ssa;
         elem_call->params[p] = nir_src_for_ssa(def);
      }
      nir_builder_instr_insert(b, &elem_call->instr);

      v[i] = nir_build_load_deref(b, 1, bit_size, &elem_deref->def, 0);
   }

   store_cmat_deref(b, dst_deref, nir_vec(b, v, length));
}

/* Decode functions take a PhysicalStorageBuffer pointer */
static nir_deref_instr *
tensor_load_base(nir_builder *b, const struct nir_calc_tensor_info *info,
                 nir_deref_instr *deref, unsigned bit_size)
{
   if (info->decode_fnptr && nir_deref_mode_is(deref, nir_var_mem_ssbo)) {
      nir_def *addr = nir_deref_buffer_address(b, 1, 64, &deref->def);
      deref = nir_build_deref_cast(b, addr, nir_var_mem_global, deref->type,
                                   bit_size / 8);
   }
   return deref;
}

/* Loads (or decodes) the element at (row, col).  With a tensor view this
 * leaves info->clipped_if open with the cursor in its then branch.
 */
static nir_def *
tensor_load_elem(nir_builder *b, struct nir_calc_tensor_info *info,
                 nir_deref_instr *deref, nir_def *row, nir_def *col,
                 unsigned bit_size)
{
   nir_deref_instr *iter_deref = deref;
   nir_def *res = nir_calc_tensor_derefs(b, info, row, col, &iter_deref);
   if (info->decode_fnptr)
      return res;

   nir_def *val = nir_load_deref(b, iter_deref);
   if (info->clamp_value) {
      val = nir_bcsel(b, info->do_clamp,
                      nir_u2uN(b, info->clamp_value, bit_size), val);
   }
   return val;
}

/* The common case of tensor addressing: a 2D tensor without blocks or
 * decoding, a view that at most transposes, a span that is exactly the
 * matrix and nothing clamped or clipped.  Then matrix element (row, col)
 * is tensor element (t0, t1) = (row, col), or (col, row) when transposed,
 * at addr + (t0 * stride + t1) * element size.  ok says whether this holds
 * at run time, and whether addr and stride are aligned to align_elems.
 */
struct fast_tensor {
   bool transpose;
   nir_def *ok;
   nir_def *addr;
   nir_def *stride;
   /* 16-byte aligned and readable, for loads made whether or not ok */
   nir_def *safe_addr;
   /* With constant clamping: like ok, but the tile may run past the end of
    * the tensor, and lim0/lim1 are how many of its rows/columns (in tensor
    * order) are inside.  NULL without clamping.
    */
   nir_def *edge_ok;
   nir_def *lim0, *lim1;
   enum gl_access_qualifier access;
};

static bool
fast_tensor_init(nir_builder *b, const struct nir_calc_tensor_info *info,
                 nir_deref_instr *deref, unsigned rows, unsigned cols,
                 unsigned elem_bytes, unsigned align_elems,
                 struct fast_tensor *ft)
{
   if (info->decode_fnptr || info->spans->num_components != 2 ||
       !nir_deref_mode_is_one_of(deref, nir_var_mem_ssbo | nir_var_mem_global))
      return false;

   if (info->clamp_mode != NIR_TENSOR_CLAMP_UNDEFINED &&
       info->clamp_mode != NIR_TENSOR_CLAMP_CONSTANT)
      return false;

   ft->transpose = false;
   if (info->view) {
      if (info->view_has_dims)
         return false;
      if (info->view_permutations[0] == 1 && info->view_permutations[1] == 0)
         ft->transpose = true;
      else if (info->view_permutations[0] != 0 || info->view_permutations[1] != 1)
         return false;
   }

   nir_def *span0 = nir_channel(b, info->spans, 0);
   nir_def *span1 = nir_channel(b, info->spans, 1);
   nir_def *off0 = nir_channel(b, info->offsets, 0);
   nir_def *off1 = nir_channel(b, info->offsets, 1);
   nir_def *stride = nir_channel(b, info->strides, 0);

   nir_def *ok = nir_iand(b, nir_ieq_imm(b, nir_channel(b, info->block_sizes, 0), 1),
                             nir_ieq_imm(b, nir_channel(b, info->block_sizes, 1), 1));
   ok = nir_iand(b, ok, nir_ieq_imm(b, span0, ft->transpose ? cols : rows));
   ok = nir_iand(b, ok, nir_ieq_imm(b, span1, ft->transpose ? rows : cols));
   ok = nir_iand(b, ok, nir_ieq_imm(b, nir_channel(b, info->strides, 1), 1));

   if (info->view) {
      ok = nir_iand(b, ok, nir_ieq_imm(b, info->clip_row_offset, 0));
      ok = nir_iand(b, ok, nir_ieq_imm(b, info->clip_col_offset, 0));
      ok = nir_iand(b, ok, nir_uge_imm(b, info->clip_row_span, rows));
      ok = nir_iand(b, ok, nir_uge_imm(b, info->clip_col_span, cols));
   }

   nir_def *edge_ok = NULL;
   ft->lim0 = ft->lim1 = NULL;
   if (info->clamp_mode == NIR_TENSOR_CLAMP_CONSTANT) {
      ok = nir_iand(b, ok, nir_ige_imm(b, off0, 0));
      ok = nir_iand(b, ok, nir_ige_imm(b, off1, 0));
      ft->lim0 = nir_usub_sat(b, nir_channel(b, info->layout_dims, 0), off0);
      ft->lim1 = nir_usub_sat(b, nir_channel(b, info->layout_dims, 1), off1);
      /* Runs of align_elems along t1 are either all inside or all outside */
      nir_def *lim1_aligned =
         nir_ieq_imm(b, nir_iand_imm(b, ft->lim1, align_elems - 1), 0);
      edge_ok = nir_iand(b, ok, nir_ior(b, nir_uge(b, ft->lim1, span1),
                                           lim1_aligned));
      ok = nir_iand(b, ok, nir_uge(b, ft->lim0, span0));
      ok = nir_iand(b, ok, nir_uge(b, ft->lim1, span1));
   }

   nir_def *elem = nir_iadd(b, nir_imul(b, off0, stride), off1);
   nir_def *base_addr = nir_deref_buffer_address(b, 1, 64, &deref->def);
   nir_def *addr = nir_iadd(b, base_addr,
                            nir_imul_imm(b, nir_u2u64(b, elem), elem_bytes));

   /* The start of the array the tensor is in is there even when the tensor
    * itself is all clamped away, and the 16 bytes around it are in the same
    * page.
    */
   nir_def *safe_addr = base_addr;
   if (deref->deref_type == nir_deref_type_array) {
      nir_deref_instr *start =
         nir_build_deref_array_imm(b, nir_deref_instr_parent(deref), 0);
      safe_addr = nir_deref_buffer_address(b, 1, 64, &start->def);
   }
   ft->safe_addr = nir_iand_imm(b, safe_addr, ~(uint64_t)15);

   if (align_elems > 1) {
      nir_def *misaligned =
         nir_ior(b, nir_iand_imm(b, nir_u2u32(b, addr), align_elems * elem_bytes - 1),
                    nir_iand_imm(b, stride, align_elems - 1));
      ok = nir_iand(b, ok, nir_ieq_imm(b, misaligned, 0));
      if (edge_ok)
         edge_ok = nir_iand(b, edge_ok, nir_ieq_imm(b, misaligned, 0));
   }

   nir_variable *var = nir_deref_instr_get_variable(deref);
   ft->access = var ? var->data.access : 0;
   ft->ok = ok;
   ft->edge_ok = edge_ok;
   ft->addr = addr;
   ft->stride = stride;
   return true;
}

static nir_def *
fast_tensor_addr(nir_builder *b, const struct fast_tensor *ft,
                 nir_def *row, nir_def *col, unsigned elem_bytes)
{
   nir_def *t0 = ft->transpose ? col : row;
   nir_def *t1 = ft->transpose ? row : col;
   nir_def *elem = nir_iadd(b, nir_imul(b, t0, ft->stride), t1);
   return nir_iadd(b, ft->addr, nir_imul_imm(b, nir_u2u64(b, elem), elem_bytes));
}

/* Where this piece of a split matrix starts in the whole matrix, as
 * nir_calc_tensor_derefs() places it.
 */
static void
tensor_piece_origin(nir_builder *b, const struct nir_calc_tensor_info *info,
                    nir_def **row, nir_def **col)
{
   *row = nir_imm_int(b, info->row_imm_offset);
   *col = nir_imm_int(b, info->col_imm_offset);
   if (!info->wg)
      return;

   nir_def *sgid = nir_load_subgroup_id(b);
   if (info->desc.use != GLSL_CMAT_USE_B && info->inner_rows > 1) {
      nir_def *sg_row = nir_udiv_imm(b, sgid, info->inner_cols);
      *row = nir_iadd(b, *row, nir_imul_imm(b, sg_row, info->desc.rows));
   }
   if (info->desc.use != GLSL_CMAT_USE_A) {
      nir_def *sg_col = nir_umod_imm(b, sgid, info->inner_cols);
      *col = nir_iadd(b, *col, nir_imul_imm(b, sg_col, info->desc.cols));
   }
}

/* Loads or stores a whole fragment through a fast_tensor.  The group_size
 * runs a lane holds are contiguous in memory when they run along t1.
 */
static void
fast_tensor_load_store(nir_builder *b, const struct nir_calc_tensor_info *info,
                       const struct fast_tensor *ft, bool is_load,
                       nir_def *clamp, nir_def **vals)
{
   const struct glsl_cmat_description desc = info->desc;
   const unsigned length = get_cmat_length(desc);
   const unsigned bit_size = glsl_base_type_bit_size(desc.element_type);
   const unsigned elem_bytes = bit_size / 8;
   const unsigned gs = cmat_group_size(desc);
   const bool runs_contiguous = (desc.use == GLSL_CMAT_USE_B) == ft->transpose;
   const unsigned run = runs_contiguous ? gs : 1;

   nir_def *row0, *col0;
   tensor_piece_origin(b, info, &row0, &col0);

   nir_def *lane_id = nir_load_subgroup_invocation(b);
   nir_def *lane_row, *lane_col;
   cmat_coord(b, desc, lane_id, nir_imm_int(b, 0), &lane_row, &lane_col);
   row0 = nir_iadd(b, row0, lane_row);
   col0 = nir_iadd(b, col0, lane_col);

   for (unsigned i = 0; i < length; i += run) {
      unsigned r, c;
      cmat_coord_imm(desc, 0, i, &r, &c);
      nir_def *row = nir_iadd_imm(b, row0, r);
      nir_def *col = nir_iadd_imm(b, col0, c);
      nir_def *addr = fast_tensor_addr(b, ft, row, col, elem_bytes);
      const unsigned bits = run * bit_size;
      const unsigned comps = DIV_ROUND_UP(bits, 32);
      if (is_load) {
         /* Past the edge of the tensor: load somewhere safe, use the clamp
          * value.
          */
         nir_def *inside = NULL;
         if (clamp) {
            nir_def *t0 = ft->transpose ? col : row;
            nir_def *t1 = ft->transpose ? row : col;
            nir_def *t1_end = nir_iadd_imm(b, t1, run - 1);
            inside = nir_iand(b, nir_ult(b, t0, ft->lim0),
                                 nir_ult(b, t1_end, ft->lim1));
            addr = nir_bcsel(b, inside, addr, ft->safe_addr);
         }
         nir_def *v = nir_load_global(b, comps, MIN2(bits, 32), addr,
                                      .align_mul = run * elem_bytes,
                                      .access = ft->access);
         v = nir_extract_bits(b, &v, 1, 0, run, bit_size);
         for (unsigned j = 0; j < run; j++) {
            vals[i + j] = nir_channel(b, v, j);
            if (inside)
               vals[i + j] = nir_bcsel(b, inside, vals[i + j], clamp);
         }
      } else {
         nir_def *v = nir_vec(b, &vals[i], run);
         v = nir_extract_bits(b, &v, 1, 0, comps, MIN2(bits, 32));
         nir_store_global(b, v, addr, .align_mul = run * elem_bytes,
                          .access = ft->access);
      }
   }
}

static void
lower_cmat_tensor_load_store(nir_builder *b, nir_cmat_call_instr *call)
{
   const bool is_load = call->op == nir_cmat_call_op_tensor_load;
   nir_deref_instr *cmat_deref = nir_src_as_deref(call->params[!is_load]);
   const struct glsl_cmat_description desc =
      *glsl_get_cmat_description(cmat_deref->type);
   const unsigned length = get_cmat_length(desc);
   const unsigned bit_size = glsl_base_type_bit_size(desc.element_type);
   nir_deref_instr *deref = nir_src_as_deref(call->params[is_load]);

   nir_def *vals[NIR_MAX_VEC_COMPONENTS];
   nir_def *clips[NIR_MAX_VEC_COMPONENTS];
   if (is_load)
      cmat_split(b, load_cmat_src(b, call->params[4]), clips);
   else
      cmat_split(b, load_cmat_deref(b, cmat_deref), vals);

   struct nir_calc_tensor_info info = { 0 };
   nir_calc_tensor_derefs_init(b, &info, call);

   deref = tensor_load_base(b, &info, deref, bit_size);

   /* Subgroups beyond the matrix hold duplicates; don't let them store */
   nir_if *sg_guard = NULL;
   if (!is_load && info.wg) {
      sg_guard = nir_push_if(b, nir_ilt_imm(b, nir_load_subgroup_id(b),
                                            info.inner_rows * info.inner_cols));
   }

   const struct glsl_cmat_description orig_desc = nir_cmat_call_cmat_desc(call);
   struct fast_tensor ft;
   nir_if *fast_if = NULL;
   if (fast_tensor_init(b, &info, deref, orig_desc.rows, orig_desc.cols,
                        bit_size / 8, cmat_group_size(desc), &ft)) {
      /* Loads can take the tile across the edge of the tensor, too */
      const bool edge = is_load && ft.edge_ok;
      fast_if = nir_push_if(b, edge ? ft.edge_ok : ft.ok);
      nir_def *clamp = edge ? nir_u2uN(b, info.clamp_value, bit_size) : NULL;
      fast_tensor_load_store(b, &info, &ft, is_load, clamp, vals);
      nir_push_else(b, fast_if);
   }

   nir_def *fast_vals[NIR_MAX_VEC_COMPONENTS];
   if (fast_if && is_load)
      memcpy(fast_vals, vals, sizeof(vals[0]) * length);

   nir_def *lane_id = nir_load_subgroup_invocation(b);
   for (unsigned i = 0; i < length; i++) {
      nir_def *row, *col;
      cmat_coord(b, desc, lane_id, nir_imm_int(b, i), &row, &col);

      if (is_load) {
         vals[i] = tensor_load_elem(b, &info, deref, row, col, bit_size);
      } else {
         nir_deref_instr *iter_deref = deref;
         nir_calc_tensor_derefs(b, &info, row, col, &iter_deref);
         nir_push_if(b, nir_inot(b, info.do_clamp));
         nir_store_deref(b, iter_deref, vals[i], 1);
         nir_pop_if(b, NULL);
      }

      if (info.clipped_if) {
         nir_push_else(b, info.clipped_if);
         nir_pop_if(b, info.clipped_if);
         if (is_load)
            vals[i] = nir_if_phi(b, vals[i], clips[i]);
         info.clipped_if = NULL;
      }
   }

   if (fast_if) {
      nir_pop_if(b, fast_if);
      if (is_load) {
         for (unsigned i = 0; i < length; i++)
            vals[i] = nir_if_phi(b, fast_vals[i], vals[i]);
      }
   }

   if (sg_guard)
      nir_pop_if(b, sg_guard);

   if (is_load)
      store_cmat_deref(b, cmat_deref, nir_vec(b, vals, length));
}

static bool
lower_cmat_call(nir_builder *b, nir_cmat_call_instr *call)
{
   b->cursor = nir_before_instr(&call->instr);

   switch (call->op) {
   case nir_cmat_call_op_reduce:
      lower_cmat_reduce(b, call);
      break;
   case nir_cmat_call_op_reduce_finish:
      lower_cmat_reduce_finish(b, call);
      break;
   case nir_cmat_call_op_reduce_2x2:
      lower_cmat_reduce_2x2(b, call);
      break;
   case nir_cmat_call_op_per_element_op:
      lower_cmat_per_element_op(b, call);
      break;
   case nir_cmat_call_op_tensor_load:
   case nir_cmat_call_op_tensor_store:
      lower_cmat_tensor_load_store(b, call);
      break;
   default:
      return false;
   }

   nir_instr_remove(&call->instr);
   return true;
}

static bool
lower_cmat_instr(nir_builder *b,
                 nir_instr *instr,
                 struct hash_table *type_mapping,
                 const struct nak_compiler *nak)
{
   /* Remap deref types */
   if (instr->type == nir_instr_type_deref) {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      const struct glsl_type *new_type =
         remap_matrix_type(type_mapping, deref->type);

      if (new_type != deref->type) {
         deref->type = new_type;
         return true;
      } else {
         return false;
      }
   }

   if (instr->type == nir_instr_type_cmat_call)
      return lower_cmat_call(b, nir_instr_as_cmat_call(instr));

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_cmat_construct: {
      const unsigned length = get_cmat_length(cmat_src_desc(intr->src[0]));
      nir_def *r = nir_replicate(b, intr->src[1].ssa, length);

      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_load: {
      lower_cmat_load(b, intr);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_store: {
      const struct glsl_cmat_description desc = cmat_src_desc(intr->src[1]);
      const enum glsl_matrix_layout layout = nir_intrinsic_matrix_layout(intr);
      const unsigned length = get_cmat_length(desc);
      nir_def *src = load_cmat_src(b, intr->src[1]);

      nir_def *vars[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < length; i++)
         vars[i] = nir_channel(b, src, i);

      nir_def *lane_id = nir_load_subgroup_invocation(b);

      int vec_size = load_store_get_vec_size(desc, layout);
      for (unsigned idx = 0; idx < length; idx += vec_size) {
         nir_deref_instr *iter_deref =
            get_cmat_component_deref(b, intr, lane_id, idx);

         nir_variable_mode modes = iter_deref->modes;
         const glsl_type *vec_type = glsl_vector_type(desc.element_type, vec_size);
         iter_deref = nir_build_deref_cast_with_alignment(b,
            &iter_deref->def, modes, vec_type,
            0, vec_size * glsl_base_type_bit_size(desc.element_type) / 8, 0);
         nir_def *value = nir_vec(b, &vars[idx], vec_size);
         if (transpose_on_load_store(desc, layout))
            value = transpose_matrix(b, value);
         nir_store_deref(b, iter_deref, value, -1);
      }

      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_length: {
      const unsigned length = get_cmat_length(nir_intrinsic_cmat_desc(intr));
      nir_def_replace(&intr->def, nir_imm_int(b, length));
      return true;
   }

   case nir_intrinsic_cmat_muladd: {
      const struct glsl_cmat_description d_desc = cmat_src_desc(intr->src[0]);

      const struct glsl_cmat_description a_desc = cmat_src_desc(intr->src[1]);
      const struct glsl_cmat_description b_desc = cmat_src_desc(intr->src[2]);
      const struct glsl_cmat_description c_desc = cmat_src_desc(intr->src[3]);

      nir_def *cmat_a = load_cmat_src(b, intr->src[1]);
      nir_def *cmat_b = load_cmat_src(b, intr->src[2]);
      nir_def *cmat_c = load_cmat_src(b, intr->src[3]);

      nir_def *ret = lower_cmat_muladd(b, intr, cmat_a, cmat_b, cmat_c, a_desc,
                                       b_desc, c_desc, d_desc, nak->sm);
      store_cmat_src(b, intr->src[0], ret);
      nir_instr_remove(&intr->instr);
      return true;
   }

   case nir_intrinsic_cmat_unary_op: {
      nir_def *src = load_cmat_src(b, intr->src[1]);
      nir_op op = nir_intrinsic_alu_op(intr);
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);

      nir_def *ret = nir_build_alu1(b, op, src);
      store_cmat_src(b, intr->src[0], ret);

      nir_instr_remove(instr);
      b->fp_math_ctrl = nir_fp_fast_math;
      return true;
   }

   case nir_intrinsic_cmat_binary_op: {
      nir_def *src_a = load_cmat_src(b, intr->src[1]);
      nir_def *src_b = load_cmat_src(b, intr->src[2]);
      nir_op op = nir_intrinsic_alu_op(intr);
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);

      nir_def *ret = nir_build_alu2(b, op, src_a, src_b);
      store_cmat_src(b, intr->src[0], ret);

      nir_instr_remove(instr);
      b->fp_math_ctrl = nir_fp_fast_math;
      return true;
   }

   case nir_intrinsic_cmat_scalar_op: {
      nir_def *src_a = load_cmat_src(b, intr->src[1]);
      nir_op op = nir_intrinsic_alu_op(intr);
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);

      nir_def *ret = nir_build_alu2(b, op, src_a, intr->src[2].ssa);
      store_cmat_src(b, intr->src[0], ret);

      nir_instr_remove(instr);
      b->fp_math_ctrl = nir_fp_fast_math;
      return true;
   }

   case nir_intrinsic_cmat_bitcast: {
      nir_def *mat = load_cmat_src(b, intr->src[1]);
      store_cmat_src(b, intr->src[0], mat);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_extract: {
      nir_def *mat = load_cmat_src(b, intr->src[0]);
      nir_def *index = intr->src[1].ssa;
      nir_def *elem = nir_vector_extract(b, mat, index);
      nir_def_replace(&intr->def, elem);
      return true;
   }

   case nir_intrinsic_cmat_insert: {
      nir_def *elem = intr->src[1].ssa;
      nir_def *mat = load_cmat_src(b, intr->src[2]);
      nir_def *index = intr->src[3].ssa;

      nir_def *r = nir_vector_insert(b, mat, elem, index);
      store_cmat_src(b, intr->src[0], r);

      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_copy: {
      nir_build_copy_deref(b, intr->src[0].ssa, intr->src[1].ssa);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_convert: {
      struct glsl_cmat_description dst_desc = cmat_src_desc(intr->src[0]);
      struct glsl_cmat_description src_desc = cmat_src_desc(intr->src[1]);
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);

      nir_def *cmat = load_cmat_src(b, intr->src[1]);
      nir_def *ret = src_desc.use == dst_desc.use ?
         lower_cmat_convert(b, intr, cmat, src_desc, dst_desc) :
         lower_cmat_relayout(b, intr, cmat, src_desc, dst_desc, false);
      store_cmat_src(b, intr->src[0], ret);

      nir_instr_remove(instr);
      b->fp_math_ctrl = nir_fp_fast_math;
      return true;
   }

   case nir_intrinsic_cmat_transpose: {
      struct glsl_cmat_description dst_desc = cmat_src_desc(intr->src[0]);
      struct glsl_cmat_description src_desc = cmat_src_desc(intr->src[1]);
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);

      nir_def *cmat = load_cmat_src(b, intr->src[1]);
      nir_def *ret = lower_cmat_relayout(b, intr, cmat, src_desc, dst_desc, true);
      store_cmat_src(b, intr->src[0], ret);

      nir_instr_remove(instr);
      b->fp_math_ctrl = nir_fp_fast_math;
      return true;
   }

   case nir_intrinsic_cmat_get_coordinate: {
      struct glsl_cmat_description desc = nir_intrinsic_cmat_desc(intr);
      nir_def *row, *col;
      cmat_coord(b, desc, nir_load_subgroup_invocation(b), intr->src[0].ssa,
                 &row, &col);
      nir_def_replace(&intr->def, nir_vec2(b, row, col));
      return true;
   }

   default:
      return false;
   }
}

static bool
lower_cmat_impl(nir_function_impl *impl,
                struct hash_table *type_mapping,
                const struct nak_compiler *nak)
{
   bool progress = false;

   /* Remap all cmat temp var to array of scalars */
   nir_foreach_function_temp_variable(var, impl) {
      const struct glsl_type *new_type =
         remap_matrix_type(type_mapping, var->type);
      if (new_type != var->type) {
         var->type = new_type;
         progress = true;
      }
   }

   /* Tensor loads and stores add control flow */
   bool added_cf = false;
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block_reverse_safe(block, impl) {
      nir_foreach_instr_reverse_safe(instr, block) {
         const bool is_call = instr->type == nir_instr_type_cmat_call;
         if (lower_cmat_instr(&b, instr, type_mapping, nak)) {
            progress = true;
            added_cf |= is_call;
         }
      }
   }

   return nir_progress(progress, impl, added_cf ? nir_metadata_none
                                                : nir_metadata_control_flow);
}

bool
nak_nir_lower_cmat(nir_shader *nir, const struct nak_compiler *nak)
{
   bool progress = false;

   if (nir->info.stage != MESA_SHADER_COMPUTE ||
       !nir->info.cs.has_cooperative_matrix)
      return false;

   struct hash_table *type_mapping = _mesa_pointer_hash_table_create(NULL);

   /* Remap all cmat shader temp var to array of scalars */
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_temp) {
      const struct glsl_type *new_type =
         remap_matrix_type(type_mapping, var->type);

      if (new_type != var->type) {
         var->type = new_type;
         progress = true;
      }
   }

   nir_foreach_function_impl(impl, nir) {
      if (lower_cmat_impl(impl, type_mapping, nak))
         progress = true;
   }

   _mesa_hash_table_destroy(type_mapping, NULL);
   return progress;
}

/* The shared memory a workgroup can have, which is more than NVK lets the
 * shader itself declare (maxComputeSharedMemorySize is 48 KiB).
 */
static unsigned
nak_cmat_max_shared_bytes(const struct nak_compiler *nak)
{
   /* nak_shader_info::cs.smem_size is 16 bits */
   const unsigned info_max = 0xff00;
   if (nak->sm >= 80)
      return MIN2(info_max, 96 * 1024); /* of 100 KiB per SM */
   else if (nak->sm >= 75)
      return MIN2(info_max, 64 * 1024);
   else
      return MIN2(info_max, 96 * 1024);
}

/* Pad rows (columns for B) by 16 bytes so that the eight 16-byte rows an
 * LDSM reads land in different banks.
 */
#define NAK_CMAT_STAGE_PAD 8

/* A is staged row-major and B column-major, which are the layouts LDSM
 * loads them from.  Consecutive invocations fill the inner dimension.
 */
struct stage_layout {
   bool col_major;
   unsigned inner;
   unsigned outer;
   unsigned stride;
   enum glsl_matrix_layout layout;
};

static struct glsl_cmat_description
stage_call_desc(nir_cmat_call_instr *call)
{
   nir_deref_instr *dst = nir_src_as_deref(call->params[0]);
   return *glsl_get_cmat_description(dst->type);
}

static struct stage_layout
get_stage_layout(nir_cmat_call_instr *call)
{
   /* B goes column-major when the tensor is read transposed, as for
    * K^T and the B of matmuls, and otherwise row-major like the tensor
    * (LDSM transposes either way), so that fills can copy whole rows.
    */
   const struct glsl_cmat_description desc = stage_call_desc(call);
   const struct nir_cmat_tensor_load tl = nir_cmat_call_tensor_load_info(call);
   const bool transposed = tl.tensor_view && tl.view_permutations[0] == 1;
   const bool col_major = desc.use == GLSL_CMAT_USE_B && transposed;
   return (struct stage_layout) {
      .col_major = col_major,
      .inner = col_major ? desc.rows : desc.cols,
      .outer = col_major ? desc.cols : desc.rows,
      .stride = (col_major ? desc.rows : desc.cols) + NAK_CMAT_STAGE_PAD,
      .layout = col_major ? GLSL_MATRIX_LAYOUT_COLUMN_MAJOR :
                            GLSL_MATRIX_LAYOUT_ROW_MAJOR,
   };
}

static unsigned
staged_tile_elems(nir_cmat_call_instr *call)
{
   const struct stage_layout l = get_stage_layout(call);
   return l.outer * l.stride;
}

static void
stage_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared);
}

/* Pointer to the fp16 element offset of the tile */
static nir_deref_instr *
stage_tile_ptr(nir_builder *b, nir_variable *tile, nir_def *offset)
{
   nir_deref_instr *deref = nir_build_deref_var(b, tile);
   deref = nir_build_deref_cast_with_alignment(b, &deref->def, nir_var_mem_shared,
                                               glsl_float16_t_type(), 2, 16, 0);
   return nir_build_deref_ptr_as_array(b, deref,
                                       nir_u2uN(b, offset, deref->def.bit_size));
}

/* Most 16-byte loads per invocation in a fast fill */
#define STAGE_MAX_FAST_LOADS 64

/* One tensor load the staging pass handles: either filled into the tile, or
 * (direct) loaded from memory into the matrix as usual.
 */
struct stage_load {
   nir_cmat_call_instr *call;
   struct nir_calc_tensor_info info;
   nir_deref_instr *base;
   struct fast_tensor ft;
   bool fast;
   unsigned offset;
};

static void
stage_load_init(nir_builder *b, struct stage_load *sl, nir_cmat_call_instr *call,
                unsigned offset, bool direct, unsigned wg_size)
{
   const struct glsl_cmat_description desc = stage_call_desc(call);
   const unsigned elem_bytes = glsl_base_type_bit_size(desc.element_type) / 8;
   const struct stage_layout l = get_stage_layout(call);

   memset(sl, 0, sizeof(*sl));
   sl->call = call;
   sl->offset = offset;
   nir_calc_tensor_derefs_init(b, &sl->info, call);
   sl->base = tensor_load_base(b, &sl->info, nir_src_as_deref(call->params[1]),
                               elem_bytes * 8);

   /* 16-byte accesses: along the inner dimension of the tile for fills,
    * and whatever cmat_load assumes for direct loads.
    */
   const unsigned vec = 16 / elem_bytes;
   if (direct) {
      sl->fast = fast_tensor_init(b, &sl->info, sl->base, desc.rows, desc.cols,
                                  elem_bytes, vec, &sl->ft);
   } else {
      sl->fast = l.inner % vec == 0 &&
                 DIV_ROUND_UP(l.inner * l.outer / vec, wg_size) <= STAGE_MAX_FAST_LOADS &&
                 fast_tensor_init(b, &sl->info, sl->base, desc.rows, desc.cols,
                                  elem_bytes, vec, &sl->ft) &&
                 sl->ft.transpose == l.col_major;
   }
}

/* The 16-byte loads of a fast fill.  They are made whatever ok is, from a
 * safe address when it's false, so that they need no branch.
 */
static void
stage_fill_fast_loads(nir_builder *b, struct stage_load *sl, nir_def *ok,
                      unsigned wg_size, nir_def **vals)
{
   const struct glsl_cmat_description desc = stage_call_desc(sl->call);
   const unsigned elem_bytes = glsl_base_type_bit_size(desc.element_type) / 8;
   const struct stage_layout l = get_stage_layout(sl->call);
   const unsigned vec = 16 / elem_bytes;
   const unsigned row_vecs = l.inner / vec;
   const unsigned total_vecs = l.inner * l.outer / vec;
   nir_def *li = nir_load_local_invocation_index(b);

   /* With whole rows per step, an invocation's loads are a constant number
    * of rows apart: pick the safe address once, and step by a stride that
    * is 0 when ok is false.  The steps are uniform, so each load costs one
    * 64-bit add instead of an address and a select.
    */
   if (wg_size % row_vecs == 0 && total_vecs % wg_size == 0) {
      const unsigned step_rows = wg_size / row_vecs;
      nir_def *in = nir_imul_imm(b, nir_iand_imm(b, li, row_vecs - 1), vec);
      nir_def *out = nir_ushr_imm(b, li, util_logbase2(row_vecs));
      nir_def *addr0 = fast_tensor_addr(b, &sl->ft, l.col_major ? in : out,
                                        l.col_major ? out : in, elem_bytes);
      addr0 = nir_bcsel(b, ok, addr0, sl->ft.safe_addr);
      nir_def *stride = nir_bcsel(b, ok, sl->ft.stride, nir_imm_int(b, 0));
      for (unsigned n = 0; n < total_vecs / wg_size; n++) {
         nir_def *off = nir_u2u64(b, nir_imul_imm(b, stride, n * step_rows));
         nir_def *addr = nir_iadd(b, addr0, nir_imul_imm(b, off, elem_bytes));
         vals[n] = nir_load_global(b, 4, 32, addr, .align_mul = 16,
                                   .access = sl->ft.access);
      }
      return;
   }

   for (unsigned v0 = 0, n = 0; v0 < total_vecs; v0 += wg_size, n++) {
      nir_def *v = nir_iadd_imm(b, li, v0);
      if (v0 + wg_size > total_vecs)
         v = nir_umin(b, v, nir_imm_int(b, total_vecs - 1));

      nir_def *in_vec = nir_iand_imm(b, v, row_vecs - 1);
      nir_def *out = nir_ushr_imm(b, v, util_logbase2(row_vecs));
      nir_def *in = nir_imul_imm(b, in_vec, vec);
      nir_def *addr = fast_tensor_addr(b, &sl->ft, l.col_major ? in : out,
                                       l.col_major ? out : in, elem_bytes);
      addr = nir_bcsel(b, ok, addr, sl->ft.safe_addr);
      vals[n] = nir_load_global(b, 4, 32, addr, .align_mul = 16,
                                .access = sl->ft.access);
   }
}

static void
stage_fill_fast_stores(nir_builder *b, struct stage_load *sl, nir_variable *tile,
                       unsigned wg_size, nir_def **vals)
{
   const struct glsl_cmat_description desc = stage_call_desc(sl->call);
   const unsigned elem_bytes = glsl_base_type_bit_size(desc.element_type) / 8;
   const struct stage_layout l = get_stage_layout(sl->call);
   const unsigned vec = 16 / elem_bytes;
   const unsigned row_vecs = l.inner / vec;
   const unsigned total_vecs = l.inner * l.outer / vec;
   nir_def *li = nir_load_local_invocation_index(b);

   nir_deref_instr *tile_vec = nir_build_deref_var(b, tile);
   tile_vec = nir_build_deref_cast_with_alignment(b, &tile_vec->def,
                                                  nir_var_mem_shared,
                                                  glsl_uvec4_type(), 16, 16, 0);

   for (unsigned v0 = 0, n = 0; v0 < total_vecs; v0 += wg_size, n++) {
      nir_def *v = nir_iadd_imm(b, li, v0);
      nir_if *tail = NULL;
      if (v0 + wg_size > total_vecs)
         tail = nir_push_if(b, nir_ult_imm(b, v, total_vecs));

      nir_def *in_vec = nir_iand_imm(b, v, row_vecs - 1);
      nir_def *out = nir_ushr_imm(b, v, util_logbase2(row_vecs));
      nir_def *idx = nir_iadd(b, nir_imul_imm(b, out, l.stride / vec), in_vec);
      idx = nir_iadd_imm(b, idx, sl->offset / vec);
      nir_store_deref(b, nir_build_deref_ptr_as_array(b, tile_vec,
                            nir_u2uN(b, idx, tile_vec->def.bit_size)),
                      vals[n], 0xf);

      if (tail)
         nir_pop_if(b, tail);
   }
}

struct stage_generic_iter {
   struct nir_calc_tensor_info *info;
   nir_deref_instr *base;
   nir_variable *tile;
   nir_def *li;
   /* Where the first run of the invocation is */
   nir_def *in0, *out0;
   unsigned wg_size;
   unsigned offset;
   unsigned run;
   unsigned unroll;
   unsigned runs;
   bool exact;
   bool vec_decode;
};

/* One iteration of a generic fill: unroll runs of run elements each.  With
 * vals_out it only computes their values, and with vals_in it only stores
 * them, counting them in *n.
 */
static void
stage_fill_generic_iter(nir_builder *b, struct stage_load *sl,
                        const struct stage_generic_iter *it, nir_def *j,
                        nir_def **vals_out, nir_def *const *vals_in,
                        unsigned *n)
{
   const struct glsl_cmat_description desc = stage_call_desc(sl->call);
   const unsigned bit_size = glsl_base_type_bit_size(desc.element_type);
   const unsigned elem_bytes = bit_size / 8;
   const struct stage_layout l = get_stage_layout(sl->call);
   struct nir_calc_tensor_info *info = it->info;
   nir_deref_instr *base = it->base;
   nir_variable *tile = it->tile;
   nir_def *li = it->li;
   const unsigned wg_size = it->wg_size, offset = it->offset, run = it->run;
   const unsigned unroll = it->unroll, runs = it->runs;
   const bool exact = it->exact, vec_decode = it->vec_decode;

   /* With whole lines per step, every run of an invocation is in the same
    * column, a constant number of lines after its first.
    */
   const bool lines = exact && nir_src_is_const(nir_src_for_ssa(j)) &&
                      (wg_size * run) % l.inner == 0;
   nir_def *in0 = it->in0, *out0 = it->out0;

   for (unsigned u = 0; u < unroll; u++) {
      nir_def *in, *out;
      unsigned out_delta = 0;
      nir_if *tail = NULL;
      if (lines) {
         out_delta = (nir_src_as_uint(nir_src_for_ssa(j)) * unroll + u) * (wg_size * run / l.inner);
         in = in0;
         out = nir_iadd_imm(b, out0, out_delta);
      } else {
         nir_def *r = nir_iadd(b, li, nir_imul_imm(b, nir_iadd_imm(b, nir_imul_imm(b, j, unroll), u),
                                                   wg_size));
         tail = exact ? NULL : nir_push_if(b, nir_ult_imm(b, r, runs));
         nir_def *e = nir_imul_imm(b, r, run);
         in = nir_iand_imm(b, e, l.inner - 1);
         out = nir_ushr_imm(b, e, util_logbase2(l.inner));
      }
      nir_def *row = l.col_major ? in : out;
      nir_def *col = l.col_major ? out : in;

      nir_def *off = nir_iadd_imm(b, nir_iadd(b, nir_imul_imm(b, out, l.stride), in),
                                  offset);
      nir_deref_instr *dst = stage_tile_ptr(b, tile, off);
      nir_deref_instr *vec_dst =
         nir_build_deref_cast_with_alignment(b, &dst->def, nir_var_mem_shared,
                                             glsl_vector_type(desc.element_type, run),
                                             run * elem_bytes, run * elem_bytes, 0);
      if (vals_in) {
         nir_store_deref(b, vec_dst, vals_in[(*n)++], nir_component_mask(run));
      } else if (vec_decode) {
         nir_deref_instr *iter_deref = base;
         info->use_vec_decode = true;
         nir_def *val = lines && !l.col_major ?
            nir_calc_tensor_decode_rows(b, info, base, out0, out_delta, in0) :
            nir_calc_tensor_derefs(b, info, row, col, &iter_deref);
         info->use_vec_decode = false;
         if (vals_out)
            vals_out[(*n)++] = val;
         else
            nir_store_deref(b, vec_dst, val, nir_component_mask(run));
         if (info->clipped_if) {
            nir_pop_if(b, info->clipped_if);
            info->clipped_if = NULL;
         }
      } else {
         /* Consecutive elements mostly share a block, which lets the
          * inlined decode functions share their work.
          */
         nir_def *vals[8];
         for (unsigned k = 0; k < run; k++) {
            nir_def *k_row = l.col_major ? nir_iadd_imm(b, row, k) : row;
            nir_def *k_col = l.col_major ? col : nir_iadd_imm(b, col, k);
            vals[k] = tensor_load_elem(b, info, base, k_row, k_col, bit_size);
            if (info->clipped_if) {
               nir_store_deref(b, nir_build_deref_ptr_as_array(b, dst,
                                     nir_imm_intN_t(b, k, dst->def.bit_size)),
                               vals[k], 1);
               nir_pop_if(b, info->clipped_if);
               info->clipped_if = NULL;
               vals[k] = NULL;
            }
         }
         if (vals_out)
            vals_out[(*n)++] = nir_vec(b, vals, run);
         else if (vals[0])
            nir_store_deref(b, vec_dst, nir_vec(b, vals, run), nir_component_mask(run));
      }
      if (tail)
         nir_pop_if(b, tail);
   }

}

/* Most runs per invocation whose values a generic fill computes ahead */
#define STAGE_MAX_PREFETCH_RUNS 64

/* How a generic fill goes: element by element, or by decode vector when the
 * vector runs along the inner dimension.
 */
struct stage_generic_plan {
   struct stage_generic_iter it;
   unsigned iters;
   bool unrolled;
   /* All values can be computed before the barrier and stored after */
   bool prefetch;
};

static void
stage_generic_plan_init(nir_builder *b, struct stage_load *sl, nir_variable *tile,
                        unsigned wg_size, struct stage_generic_plan *plan)
{
   const struct glsl_cmat_description desc = stage_call_desc(sl->call);
   const struct nir_cmat_tensor_load tl = nir_cmat_call_tensor_load_info(sl->call);
   const struct stage_layout l = get_stage_layout(sl->call);
   const unsigned total = l.inner * l.outer;
   struct nir_calc_tensor_info *info = &sl->info;

   unsigned run = l.inner % 8 == 0 ? 8 : 1;
   bool vec_decode = false;
   if (info->decode_vec_fnptr) {
      const bool transposed = info->view && info->view_permutations[0] == 1;
      const unsigned width = glsl_get_vector_elements(
         glsl_get_bare_type(info->decode_vec_fnptr->params[0].type));
      if (!info->view_has_dims && transposed == l.col_major &&
          width > 1 && width <= 8 && l.inner % width == 0) {
         run = width;
         vec_decode = true;
      }
   }

   /* Each invocation handles per_inv runs, about 8 elements of them per
    * iteration so that their loads overlap without running out of
    * registers.  The trip count is the same everywhere.
    */
   const unsigned runs = total / run;
   const unsigned per_inv = DIV_ROUND_UP(runs, wg_size);
   const unsigned unroll = MIN2(per_inv, MAX2(1, 8 / run));
   const bool exact = runs % (wg_size * unroll) == 0;
   plan->iters = DIV_ROUND_UP(per_inv, unroll);

   /* Inside a loop, keep it the innermost one so that it can be unrolled:
    * shaders unroll their K loops, and that is what lets the loads of one
    * step move up past the multiplies of the step before.
    */
   bool in_loop = false;
   for (nir_cf_node *n = sl->call->instr.block->cf_node.parent; n; n = n->parent)
      in_loop |= n->type == nir_cf_node_loop;
   plan->unrolled = in_loop || plan->iters == 1;

   plan->prefetch = plan->unrolled && exact && !tl.tensor_view &&
                    plan->iters * unroll <= STAGE_MAX_PREFETCH_RUNS;

   nir_def *li = nir_load_local_invocation_index(b);
   nir_def *e0 = nir_imul_imm(b, li, run);
   plan->it = (struct stage_generic_iter) {
      .info = info, .base = sl->base, .tile = tile, .li = li,
      .in0 = nir_iand_imm(b, e0, l.inner - 1),
      .out0 = nir_ushr_imm(b, e0, util_logbase2(l.inner)),
      .wg_size = wg_size,
      .offset = sl->offset, .run = run, .unroll = unroll, .runs = runs,
      .exact = exact, .vec_decode = vec_decode,
   };
}

/* Computes the values of a prefetch plan, for stage_generic_stores() */
static void
stage_generic_values(nir_builder *b, struct stage_load *sl,
                     const struct stage_generic_plan *plan, nir_def **vals)
{
   unsigned n = 0;
   sl->info.row_base.row = NULL;
   for (unsigned j = 0; j < plan->iters; j++)
      stage_fill_generic_iter(b, sl, &plan->it, nir_imm_int(b, j), vals, NULL, &n);
}

static void
stage_generic_stores(nir_builder *b, struct stage_load *sl,
                     const struct stage_generic_plan *plan, nir_def *const *vals)
{
   unsigned n = 0;
   for (unsigned j = 0; j < plan->iters; j++)
      stage_fill_generic_iter(b, sl, &plan->it, nir_imm_int(b, j), NULL, vals, &n);
}

/* Fills the tile however the load needs it */
static void
stage_fill_generic(nir_builder *b, struct stage_load *sl, nir_variable *tile,
                   unsigned wg_size)
{
   nir_cmat_call_instr *call = sl->call;
   const struct glsl_cmat_description desc = stage_call_desc(call);
   const struct nir_cmat_tensor_load tl = nir_cmat_call_tensor_load_info(call);
   const struct stage_layout l = get_stage_layout(sl->call);

   /* Clipped elements come from the clip matrix.  Put it in the tile first
    * unless the view can't clip.
    */
   if (tl.tensor_view) {
      struct nir_calc_tensor_info *info = &sl->info;
      nir_def *no_clip =
         nir_iand(b, nir_iand(b, nir_ieq_imm(b, info->clip_row_offset, 0),
                                 nir_ieq_imm(b, info->clip_col_offset, 0)),
                     nir_iand(b, nir_uge_imm(b, info->clip_row_span, desc.rows),
                                 nir_uge_imm(b, info->clip_col_span, desc.cols)));
      nir_push_if(b, nir_inot(b, no_clip));
      nir_cmat_store(b, &stage_tile_ptr(b, tile, nir_imm_int(b, sl->offset))->def,
                     call->params[4].ssa, nir_imm_int(b, l.stride),
                     .matrix_layout = l.layout);
      stage_barrier(b);
      nir_pop_if(b, NULL);
   }

   struct stage_generic_plan plan;
   stage_generic_plan_init(b, sl, tile, wg_size, &plan);

   if (plan.unrolled) {
      sl->info.row_base.row = NULL;
      for (unsigned j = 0; j < plan.iters; j++)
         stage_fill_generic_iter(b, sl, &plan.it, nir_imm_int(b, j), NULL, NULL, NULL);
   } else {
      nir_variable *j_var =
         nir_local_variable_create(b->impl, glsl_uint_type(), "cmat_stage_j");
      nir_store_var(b, j_var, nir_imm_int(b, 0), 1);
      nir_loop *loop = nir_push_loop(b);
      {
         nir_def *j = nir_load_var(b, j_var);
         nir_break_if(b, nir_uge_imm(b, j, plan.iters));
         stage_fill_generic_iter(b, sl, &plan.it, j, NULL, NULL, NULL);
         nir_store_var(b, j_var, nir_iadd_imm(b, j, 1), 1);
      }
      nir_pop_loop(b, loop);
   }
}

/* The direct load as a plain strided load from memory */
static void
stage_direct_load_fast(nir_builder *b, struct stage_load *sl)
{
   const struct glsl_cmat_description desc = stage_call_desc(sl->call);
   const unsigned elem_bytes = glsl_base_type_bit_size(desc.element_type) / 8;

   nir_deref_instr *ptr =
      nir_build_deref_cast_with_alignment(b, sl->ft.addr, nir_var_mem_global,
                                          glsl_scalar_type(desc.element_type),
                                          elem_bytes, 16, 0);
   nir_cmat_load(b, sl->call->params[0].ssa, &ptr->def, sl->ft.stride,
                 .matrix_layout = sl->ft.transpose ? GLSL_MATRIX_LAYOUT_COLUMN_MAJOR :
                                                     GLSL_MATRIX_LAYOUT_ROW_MAJOR);
}

static void
stage_load_from_tile(nir_builder *b, nir_cmat_call_instr *call,
                     nir_variable *tile, unsigned offset)
{
   const struct stage_layout l = get_stage_layout(call);
   nir_cmat_load(b, call->params[0].ssa,
                 &stage_tile_ptr(b, tile, nir_imm_int(b, offset))->def,
                 nir_imm_int(b, l.stride), .matrix_layout = l.layout);
}

static bool
can_stage_tensor_load(nir_cmat_call_instr *call)
{
   if (call->op != nir_cmat_call_op_tensor_load)
      return false;

   const struct glsl_cmat_description desc = stage_call_desc(call);
   const struct stage_layout l = get_stage_layout(call);

   return desc.scope == SCOPE_WORKGROUP &&
          desc.use != GLSL_CMAT_USE_ACCUMULATOR &&
          desc.element_type == GLSL_TYPE_FLOAT16 &&
          util_is_power_of_two_nonzero(l.inner);
}

/* For nir_foreach_src(): false if src is a deref of the variable data */
static bool
src_is_var_deref(nir_src *src, void *data)
{
   nir_deref_instr *deref = nir_src_as_deref(*src);
   return !(deref && nir_deref_instr_get_variable(deref) == data);
}

#define STAGE_MAX_COPIES 8

/* Whether the fill of first can move down to second: nothing in between may
 * synchronize or look at first's result, except copies of it (vtn's
 * temporaries) that nothing else in between uses.  Those are returned, to
 * move after the staging.
 */
static bool
can_pair_stage(nir_cmat_call_instr *first, nir_cmat_call_instr *second,
               nir_instr **copies, unsigned *num_copies)
{
   *num_copies = 0;
   if (first->instr.block != second->instr.block)
      return false;

   nir_variable *vars[1 + STAGE_MAX_COPIES];
   unsigned num_vars = 0;
   vars[num_vars] = nir_deref_instr_get_variable(nir_src_as_deref(first->params[0]));
   if (!vars[num_vars++])
      return false;

   for (nir_instr *instr = nir_instr_next(&first->instr);
        instr != &second->instr; instr = nir_instr_next(instr)) {
      switch (instr->type) {
      case nir_instr_type_deref:
         continue;
      case nir_instr_type_call:
      case nir_instr_type_cmat_call:
         return false;
      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_barrier)
            return false;
         if (intr->intrinsic == nir_intrinsic_cmat_copy) {
            nir_variable *src = nir_deref_instr_get_variable(nir_src_as_deref(intr->src[1]));
            nir_variable *dst = nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));
            bool reads_result = false;
            for (unsigned v = 0; v < num_vars; v++)
               reads_result |= src == vars[v];
            if (reads_result) {
               if (!dst || *num_copies == STAGE_MAX_COPIES)
                  return false;
               copies[(*num_copies)++] = instr;
               vars[num_vars++] = dst;
               continue;
            }
         }
         break;
      }
      default:
         break;
      }

      for (unsigned v = 0; v < num_vars; v++) {
         if (!nir_foreach_src(instr, src_is_var_deref, vars[v]))
            return false;
      }
   }

   for (unsigned v = 0; v < num_vars; v++) {
      if (!nir_foreach_src(&second->instr, src_is_var_deref, vars[v]))
         return false;
   }
   return true;
}

/* Whether the whole staging of first can move down to after second: on
 * top of can_pair_stage(), nothing in between may write what first reads.
 */
static bool
can_defer_stage(nir_cmat_call_instr *first, nir_cmat_call_instr *second,
                nir_instr **copies, unsigned *num_copies)
{
   if (!can_pair_stage(first, second, copies, num_copies))
      return false;

   for (nir_instr *instr = nir_instr_next(&first->instr);
        instr != &second->instr; instr = nir_instr_next(instr)) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_store_deref &&
          intr->intrinsic != nir_intrinsic_copy_deref)
         continue;
      nir_variable *var = nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));
      if (!var || !nir_foreach_src(&first->instr, src_is_var_deref, var))
         return false;
   }
   return true;
}

/* The next tensor load in the same block, if any */
static nir_cmat_call_instr *
next_tensor_load(nir_cmat_call_instr *call)
{
   for (nir_instr *instr = nir_instr_next(&call->instr); instr;
        instr = nir_instr_next(instr)) {
      if (instr->type == nir_instr_type_cmat_call &&
          nir_instr_as_cmat_call(instr)->op == nir_cmat_call_op_tensor_load)
         return nir_instr_as_cmat_call(instr);
   }
   return NULL;
}

struct stage_group {
   nir_cmat_call_instr *calls[2];
   unsigned count;
   /* Unstaged tensor load to stage after, so that its global loads
    * overlap with the fill's.
    */
   nir_cmat_call_instr *after;
   /* Copies of the first load's result to move after the staging */
   nir_instr *copies[STAGE_MAX_COPIES];
   unsigned num_copies;
};

/* Workgroup scope matrices are split into one piece per subgroup, and each
 * subgroup would load (and decode) every A or B element it needs on its
 * own, straight from memory.  Instead, have the whole workgroup load each
 * element once into shared memory and let the subgroups load their pieces
 * from there.  Split by columns, every subgroup needs all of A but only
 * its own columns of B, so B is only staged together with an A load right
 * before it, sharing its barriers; that still makes its loads 16 bytes
 * wide.  Split by rows, it is the other way around.  Workgroup scope loads are
 * in workgroup uniform control flow.
 *
 * The tile is returned so that the split pass can share it for its
 * temporaries: each use of either starts with a barrier.
 */
bool
nak_nir_stage_cmat_tensor_loads(nir_shader *nir, const struct nak_compiler *nak,
                                unsigned inner_box_rows, nir_variable **tile_out)
{
   *tile_out = NULL;
   if (nir->info.workgroup_size_variable)
      return false;

   const unsigned wg_size = nir->info.workgroup_size[0] *
                            nir->info.workgroup_size[1] *
                            nir->info.workgroup_size[2];
   /* The same subgroup box as nir_lower_cooperative_matrix_flexible_dimensions() */
   const unsigned num_sg = DIV_ROUND_UP(wg_size, NAK_SUBGROUP_SIZE);
   unsigned box_cols = MIN2(num_sg, 8);
   unsigned box_rows = num_sg / box_cols;
   if (inner_box_rows) {
      box_rows = MIN2(inner_box_rows, num_sg);
      box_cols = num_sg / box_rows;
   }
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);

   unsigned app_shared = 0;
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_shared) {
      unsigned size, align;
      glsl_get_natural_size_align_bytes(var->type, &size, &align);
      app_shared = align_uintptr(app_shared, align) + size;
   }
   const unsigned max_shared = nak_cmat_max_shared_bytes(nak);
   const unsigned budget = max_shared - MIN2(app_shared, max_shared);

   void *mem_ctx = ralloc_context(NULL);
   struct util_dynarray cands, groups;
   util_dynarray_init(&cands, mem_ctx);
   util_dynarray_init(&groups, mem_ctx);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_cmat_call)
            continue;
         nir_cmat_call_instr *call = nir_instr_as_cmat_call(instr);
         if (can_stage_tensor_load(call))
            util_dynarray_append(&cands, call);
      }
   }

   const unsigned num_cands =
      util_dynarray_num_elements(&cands, nir_cmat_call_instr *);
   nir_cmat_call_instr **cand = cands.data;
   unsigned max_bytes = 0;
   for (unsigned i = 0; i < num_cands; i++) {
      const struct glsl_cmat_description desc = stage_call_desc(cand[i]);
      const unsigned bytes = staged_tile_elems(cand[i]) * 2;
      const bool dup = desc.use == GLSL_CMAT_USE_A ? box_cols > 1 : box_rows > 1;

      if (i + 1 < num_cands) {
         const unsigned pair_bytes =
            bytes + staged_tile_elems(cand[i + 1]) * 2;
         struct stage_group g = { { cand[i], cand[i + 1] }, 2 };
         if (pair_bytes <= budget &&
             can_defer_stage(cand[i], cand[i + 1], g.copies, &g.num_copies)) {
            util_dynarray_append(&groups, g);
            max_bytes = MAX2(max_bytes, pair_bytes);
            i++;
            continue;
         }
      }

      if (dup && bytes <= budget) {
         struct stage_group g = { { cand[i], NULL }, 1 };
         util_dynarray_append(&groups, g);
         max_bytes = MAX2(max_bytes, bytes);
      }
   }

   /* A single staged load followed by one that isn't staged gets staged
    * after that one.
    */
   struct set *staged = _mesa_pointer_set_create(mem_ctx);
   util_dynarray_foreach(&groups, struct stage_group, g) {
      for (unsigned i = 0; i < g->count; i++)
         _mesa_set_add(staged, g->calls[i]);
   }
   util_dynarray_foreach(&groups, struct stage_group, g) {
      nir_cmat_call_instr *next = next_tensor_load(g->calls[0]);
      if (g->count == 1 && next && !_mesa_set_search(staged, next) &&
          can_defer_stage(g->calls[0], next, g->copies, &g->num_copies))
         g->after = next;
   }

   if (max_bytes == 0 || getenv("NAK_CMAT_NO_STAGE")) {
      ralloc_free(mem_ctx);
      nir_foreach_function_impl(fimpl, nir)
         nir_no_progress(fimpl);
      return false;
   }

   /* One tile for all loads: each fill waits for the readers of the last */
   nir_variable *tile =
      nir_variable_create(nir, nir_var_mem_shared,
                          glsl_array_type(glsl_uvec4_type(),
                                          DIV_ROUND_UP(max_bytes, 16), 16),
                          "nak_cmat_stage");

   nir_builder b = nir_builder_create(impl);
   util_dynarray_foreach(&groups, struct stage_group, g) {
      unsigned offsets[2] = { 0, 0 };
      if (g->count > 1)
         offsets[1] = staged_tile_elems(g->calls[0]);

      /* Everything goes in one place, after the last load, when the earlier
       * ones can move there.  Then one branch picks the fast path for all of
       * them: NAK waits for all outstanding loads at every branch.
       */
      const bool together = g->after || g->count > 1;
      b.cursor = g->after ? nir_after_instr(&g->after->instr) :
                 together ? nir_before_instr(&g->calls[1]->instr) :
                            nir_before_instr(&g->calls[0]->instr);

      struct stage_load sl[2], direct;
      bool all_fast = together;
      if (together) {
         for (unsigned i = 0; i < g->count; i++) {
            stage_load_init(&b, &sl[i], g->calls[i], offsets[i], false, wg_size);
            all_fast &= sl[i].fast;
         }
         if (g->after) {
            stage_load_init(&b, &direct, g->after, 0, true, wg_size);
            all_fast &= direct.fast;
         }
      }

      if (all_fast) {
         nir_def *ok = g->after ? direct.ft.ok : nir_imm_true(&b);
         for (unsigned i = 0; i < g->count; i++)
            ok = nir_iand(&b, ok, sl[i].ft.ok);

         /* The fills' global loads go first, ahead of the barrier and any
          * branch, so that they can overlap with what came before.
          */
         nir_def *vals[2][STAGE_MAX_FAST_LOADS];
         for (unsigned i = 0; i < g->count; i++)
            stage_fill_fast_loads(&b, &sl[i], ok, wg_size, vals[i]);

         /* Readers of the previous tile are done */
         stage_barrier(&b);

         nir_push_if(&b, ok);
         if (g->after)
            stage_direct_load_fast(&b, &direct);
         for (unsigned i = 0; i < g->count; i++)
            stage_fill_fast_stores(&b, &sl[i], tile, wg_size, vals[i]);
         nir_push_else(&b, NULL);
         if (g->after) {
            nir_instr_remove(&g->after->instr);
            nir_builder_instr_insert(&b, &g->after->instr);
         }
         for (unsigned i = 0; i < g->count; i++)
            stage_fill_generic(&b, &sl[i], tile, wg_size);
         nir_pop_if(&b, NULL);
      } else {
         /* Each load picks its own path, with all the global loads that
          * can go before the barrier going there.
          */
         if (!together) {
            assert(g->count == 1);
            stage_load_init(&b, &sl[0], g->calls[0], offsets[0], false, wg_size);
         }

         struct stage_generic_plan plans[2];
         nir_def *vals[2][MAX2(STAGE_MAX_FAST_LOADS, STAGE_MAX_PREFETCH_RUNS)];
         for (unsigned i = 0; i < g->count; i++) {
            stage_generic_plan_init(&b, &sl[i], tile, wg_size, &plans[i]);
            if (sl[i].fast)
               stage_fill_fast_loads(&b, &sl[i], sl[i].ft.ok, wg_size, vals[i]);
            else if (plans[i].prefetch)
               stage_generic_values(&b, &sl[i], &plans[i], vals[i]);
         }

         /* Readers of the previous tile are done */
         stage_barrier(&b);

         for (unsigned i = 0; i < g->count; i++) {
            if (sl[i].fast) {
               nir_push_if(&b, sl[i].ft.ok);
               stage_fill_fast_stores(&b, &sl[i], tile, wg_size, vals[i]);
               nir_push_else(&b, NULL);
               stage_fill_generic(&b, &sl[i], tile, wg_size);
               nir_pop_if(&b, NULL);
            } else if (plans[i].prefetch) {
               stage_generic_stores(&b, &sl[i], &plans[i], vals[i]);
            } else {
               stage_fill_generic(&b, &sl[i], tile, wg_size);
            }
         }
      }

      stage_barrier(&b);
      for (unsigned i = 0; i < g->count; i++)
         stage_load_from_tile(&b, g->calls[i], tile, offsets[i]);
      for (unsigned i = 0; i < g->num_copies; i++) {
         nir_instr_remove(g->copies[i]);
         nir_builder_instr_insert(&b, g->copies[i]);
      }
      for (unsigned i = 0; i < g->count; i++)
         nir_instr_remove(&g->calls[i]->instr);
   }

   ralloc_free(mem_ctx);
   *tile_out = tile;

   nir_foreach_function_impl(fimpl, nir) {
      if (fimpl != impl)
         nir_no_progress(fimpl);
   }
   return nir_progress(true, impl, nir_metadata_none);
}

static nir_variable *
cmat_load_dst_var(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return NULL;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic == nir_intrinsic_cmat_load) {
      /* Global loads stay together so that their latencies overlap */
      if (!nir_deref_mode_is(nir_src_as_deref(intr->src[1]), nir_var_mem_shared))
         return NULL;
   } else if (intr->intrinsic != nir_intrinsic_cmat_copy) {
      return NULL;
   }
   return nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));
}

/* Instructions a matrix load may move past: they only compute values,
 * read memory, write matrices or write local variables the load doesn't
 * read.
 */
static bool
can_sink_cmat_load_past(nir_instr *instr, nir_instr *load)
{
   switch (instr->type) {
   case nir_instr_type_alu:
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
   case nir_instr_type_deref:
      return true;
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_deref:
      case nir_intrinsic_cmat_load:
      case nir_intrinsic_cmat_copy:
      case nir_intrinsic_cmat_muladd:
         return true;
      case nir_intrinsic_store_deref: {
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         nir_variable *var = nir_deref_instr_get_variable(deref);
         return var && nir_deref_mode_is(deref, nir_var_function_temp) &&
                nir_foreach_src(load, src_is_var_deref, var);
      }
      default:
         return nir_intrinsic_infos[intr->intrinsic].flags & NIR_INTRINSIC_CAN_REORDER;
      }
   }
   case nir_instr_type_cmat_call:
      return nir_instr_as_cmat_call(instr)->op == nir_cmat_call_op_tensor_load;
   default:
      return false;
   }
}

/* Splitting loads every piece of a matrix up front and then multiplies
 * them, which keeps all the pieces live at once.  Workgroup scope A
 * matrices are whole in every subgroup, which doesn't fit the register
 * file next to the accumulators.  Load (and copy) each piece where it is
 * first used if it comes from shared memory, which is quick.  Going
 * backwards, copies move first and their sources then follow them.
 */
bool
nak_nir_sink_cmat_loads(nir_shader *nir)
{
   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      bool impl_progress = false;
      nir_foreach_block(block, impl) {
         nir_foreach_instr_reverse_safe(instr, block) {
            nir_variable *dst = cmat_load_dst_var(instr);
            if (!dst)
               continue;

            /* A copy can't pass writes of its source either */
            nir_variable *src = NULL;
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_cmat_copy)
                  src = nir_deref_instr_get_variable(nir_src_as_deref(intr->src[1]));
            }

            /* Derefs only compute addresses (and the split leaves unused
             * ones around), the users are what count.
             */
            nir_instr *target = nir_instr_next(instr);
            while (target && can_sink_cmat_load_past(target, instr) &&
                   (target->type == nir_instr_type_deref ||
                    (nir_foreach_src(target, src_is_var_deref, dst) &&
                     (!src || nir_foreach_src(target, src_is_var_deref, src)))))
               target = nir_instr_next(target);

            if (target == nir_instr_next(instr))
               continue;

            nir_instr_remove(instr);
            if (target)
               nir_instr_insert_before(target, instr);
            else
               nir_instr_insert(nir_after_block_before_jump(block), instr);
            impl_progress = true;
         }
      }
      progress |= nir_progress(impl_progress, impl,
                               nir_metadata_control_flow);
   }
   return progress;
}
