#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

// Определение экспортного спецификатора
#ifdef _WIN32
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

// Векторизованная и многопоточная функция применения оператора Собеля
DLL_EXPORT void apply_sobel_vectorized_aligned_multithreaded(unsigned char *gray_image, int width, int height, unsigned char *output_image) {
    if (!check_avx2_support()) {
        fprintf(stderr, "AVX2 не поддерживается на этом процессоре\n");
        return;
    }

    // Определение масок Собеля
    const int8_t Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int8_t Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    // Обнуление границ изображения
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        output_image[y * width + 0] = 0;
        output_image[y * width + (width - 1)] = 0;
    }

    #pragma omp parallel for
    for (int x = 0; x < width; x++) {
        output_image[0 * width + x] = 0;
        output_image[(height - 1) * width + x] = 0;
    }

    // Векторизованная обработка внутренней части изображения
    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x <= width - 33; x += 32) {
            __m256i sumX1 = _mm256_setzero_si256();
            __m256i sumY1 = _mm256_setzero_si256();
            __m256i sumX2 = _mm256_setzero_si256();
            __m256i sumY2 = _mm256_setzero_si256();

            for (int j = -1; j <= 1; j++) {
                size_t offset_m1 = (y + j) * width + x - 1;
                size_t offset_0  = (y + j) * width + x;
                size_t offset_p1 = (y + j) * width + x + 1;

                // Загрузка пикселей
                __m256i row_m1 = _mm256_loadu_si256((__m256i*)&gray_image[offset_m1]);
                __m256i row_0  = _mm256_loadu_si256((__m256i*)&gray_image[offset_0]);
                __m256i row_p1 = _mm256_loadu_si256((__m256i*)&gray_image[offset_p1]);

                __m256i pixels_m1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(row_m1));
                __m256i pixels_0  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(row_0));
                __m256i pixels_p1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(row_p1));

                __m256i kx_m1 = _mm256_set1_epi16(Gx[j + 1][0]);
                __m256i kx_0  = _mm256_set1_epi16(Gx[j + 1][1]);
                __m256i kx_p1 = _mm256_set1_epi16(Gx[j + 1][2]);
                __m256i ky_m1 = _mm256_set1_epi16(Gy[j + 1][0]);
                __m256i ky_0  = _mm256_set1_epi16(Gy[j + 1][1]);
                __m256i ky_p1 = _mm256_set1_epi16(Gy[j + 1][2]);

                // Применение масок и накопление
                sumX1 = _mm256_add_epi16(sumX1, _mm256_mullo_epi16(kx_m1, pixels_m1));
                sumX1 = _mm256_add_epi16(sumX1, _mm256_mullo_epi16(kx_0,  pixels_0));
                sumX1 = _mm256_add_epi16(sumX1, _mm256_mullo_epi16(kx_p1, pixels_p1));

                sumY1 = _mm256_add_epi16(sumY1, _mm256_mullo_epi16(ky_m1, pixels_m1));
                sumY1 = _mm256_add_epi16(sumY1, _mm256_mullo_epi16(ky_0,  pixels_0));
                sumY1 = _mm256_add_epi16(sumY1, _mm256_mullo_epi16(ky_p1, pixels_p1));
            }

            __m256i abs_sumX1 = _mm256_abs_epi16(sumX1);
            __m256i abs_sumY1 = _mm256_abs_epi16(sumY1);
            __m256i magnitude1 = _mm256_adds_epu16(abs_sumX1, abs_sumY1);

            __m256i magnitude_packed = _mm256_packus_epi16(magnitude1, magnitude1);

            // Хранение результата
            _mm256_storeu_si256((__m256i*)&output_image[y * width + x], magnitude_packed);
        }
    }

    // Скалярная обработка оставшихся пикселей
    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = (width / 32) * 32; x < width - 1; x++) {
            int sumX = 0;
            int sumY = 0;

            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    int pixel = gray_image[(y + j) * width + (x + i)];
                    sumX += Gx[j + 1][i + 1] * pixel;
                    sumY += Gy[j + 1][i + 1] * pixel;
                }
            }

            int magnitude = abs(sumX) + abs(sumY);
            output_image[y * width + x] = (magnitude > 255) ? 255 : (unsigned char)magnitude;
        }
    }
}
