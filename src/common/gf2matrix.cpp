/*
 *  Copyright 2020 Roberto Francescon
 *  Copyright 2022 Dominik Danelski
 *  This file is part of freeRaptor.
 *
 *  freeRaptor is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  freeRaptor is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with freeRaptor.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <gf2matrix.hpp>
#include <math.h>

#define min(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a < _b ? _a : _b;                                                         \
  })

uint32_t wordsize = 8 * sizeof(word);
uint32_t wordbitmask = 8 * sizeof(word);
uint32_t wordshift = 5;

void allocate_gf2matrix(gf2matrix *mat, uint32_t n_rows, uint32_t n_cols) {
    mat->n_rows = n_rows;
    mat->n_cols = n_cols;
    mat->n_words = (mat->n_cols + wordsize - 1) >> wordshift; // Number of words needed to represent a row

    mat->rows = (word **)malloc(mat->n_rows * sizeof(word *));
    for (int i = 0; i < mat->n_rows; i++) { // Change `<=` to `<`
        mat->rows[i] = (word *)malloc(mat->n_words * sizeof(word)); // Allocate enough space for words
        memset(mat->rows[i], 0, mat->n_words * sizeof(word)); // Clear the allocated memory
    }

    mat->m_data = (word *)mat->rows[0];
}

void dealloc_gf2matrix(gf2matrix *mat) {
  for (int i = 0; i <= mat->n_rows; i++)
    free(mat->rows[i]);

  free(mat->rows);
}

uint32_t get_nrows(gf2matrix *mat) { return mat->n_rows; }

uint32_t get_ncols(gf2matrix *mat) { return mat->n_cols; }

uint32_t get_nwords(gf2matrix *mat) { return mat->n_words; }

int get_entry(gf2matrix *mat, int n, int m) {
  return (mat->rows[n][(m / wordbitmask)] >> m) & 1;
}

word *get_word(gf2matrix *mat, int n, int l) {
  return mat->rows[n] + l * sizeof(word);
}

void set_entry(gf2matrix *mat, int n, int m, int val) {
  mat->rows[n][(m / wordbitmask)] ^=
      (-val ^ mat->rows[n][(m / wordbitmask)]) & (1 << m);
}

void swap_rows(gf2matrix *mat, int n, int k) {
  word *tmp = mat->rows[n];
  mat->rows[n] = mat->rows[k];
  mat->rows[k] = tmp;
}

void print_matrix(gf2matrix *mat) {
  uint32_t nrows = get_nrows(mat);
  uint32_t ncols = get_ncols(mat);

  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++)
      printf("%d ", get_entry(mat, i, j));

    printf("\n");
  }
}

void print_matrix_to_file(gf2matrix *mat) {
  printf("Print to file.\n");
  FILE *file = fopen("A.dat", "w"); // 打开文件以写入，如果文件不存在则创建
  if (file == NULL)
  {
    printf("无法打开文件\n");
    return;
  }

  uint32_t nrows = get_nrows(mat);
  uint32_t ncols = get_ncols(mat);

  for (int i = 0; i < nrows; i++)
  {
    for (int j = 0; j < ncols; j++)
      fprintf(file, "%d ", get_entry(mat, i, j));

    fprintf(file, "\n"); // 每行结束后写入换行符
  }
  fclose(file); // 关闭文件
}

void print_matrix_by_n(gf2matrix *mat, int n) {
  uint32_t nrows = get_nrows(mat);
  uint32_t ncols = get_ncols(mat);

  if(n < 5) return;

  for (int i = n -5; i < n+5; i++) {
    printf("%d: ", i);
    for (int j = n -5; j < n+5; j++)
      printf("%d(%d, %d) ", get_entry(mat, i, j),i,j);

    printf("\n");
  }
}

void swap_cols(gf2matrix *mat, int m, int k) {
  uint32_t nrows = get_nrows(mat);

  for (int i = 0; i < nrows; i++) {
    int tmp_val = get_entry(mat, i, m);
    set_entry(mat, i, m, get_entry(mat, i, k));
    set_entry(mat, i, k, tmp_val);
  }
}

void create_identity_matrix(gf2matrix *identity, uint32_t ncols,
                            uint32_t nrows) {
  allocate_gf2matrix(identity, nrows, ncols);
  for (int i = 0; i < min(ncols, nrows); i++)
    set_entry(identity, i, i, 1);
}

void mat_mul(gf2matrix *matA, gf2matrix *matB, gf2matrix *result) {
  int part;
  uint32_t ncols_A = get_ncols(matA);
  uint32_t nrows_B = get_nrows(matB);

  for (int i = 0; i < ncols_A; i++) {
    for (int j = 0; j < nrows_B; j++) {
      part = get_entry(matA, i, 0) * get_entry(matB, 0, j);

      for (int z = 1; z < ncols_A; z++)
        part = part ^ (get_entry(matA, i, z) * get_entry(matB, z, j));

      set_entry(result, i, j, part);
    }
  }
}

int gaussjordan_inv(gf2matrix *mat) {
  gf2matrix identity;
  create_identity_matrix(&identity, get_ncols(mat), get_nrows(mat));

  uint32_t nrows_mat = get_nrows(mat);
  uint32_t ncols_mat = get_ncols(mat);
  uint32_t nwords_mat = get_nwords(mat);
  for (int i = 0; i < nrows_mat; ++i) {

    int j = i;
    while (j < nrows_mat ){
      if (get_entry(mat, j, i) == 1) break;
      j++;
    };
    if(j == nrows_mat){
      printf("no pivot element, can't inverse!\n");
      return 1;
    }


    if (i != j){
      swap_rows(mat, i, j);
      swap_rows(&identity, i, j);
    }


    for (int k = 0; k < nrows_mat; k++){
      if (k != i && get_entry(mat, k, i) == 1){
        for (int l = 0; l < nwords_mat; l++) {
          mat->rows[k][l] = mat->rows[k][l] ^ mat->rows[i][l];
          identity.rows[k][l] = identity.rows[k][l] ^ identity.rows[i][l];
        }
      }
    }

    // printf("row %d\n", i);
    // printf("Found 1 on line %d\n", j);
    // char input;
    // scanf("%c", &input );
  }

  *mat = identity;
  return 0;
}
