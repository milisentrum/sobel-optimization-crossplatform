// sobel_dll.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

// Определение экспортного спецификатора
#ifdef _WIN32
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

// Векторизованная функция применения оператора Собеля
DLL_EXPORT void apply_sobel(unsigned char *gray_image, int width, int height, unsigned char *output_image) {
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
    for(int y = 0; y < height; y++) {
        output_image[y * width + 0] = 0;
        output_image[y * width + (width - 1)] = 0;
    }
    for(int x = 0; x < width; x++) {
        output_image[0 * width + x] = 0;
        output_image[(height - 1) * width + x] = 0;
    }

    // Векторизованная обработка внутренней части изображения
    for (int y = 1; y < height - 1; y++) {
        int x = 1;
        // Обработка по 32 пикселя за раз
        for (; x <= width - 33; x += 32) {
            __m256i sumX1 = _mm256_setzero_si256();
            __m256i sumY1 = _mm256_setzero_si256();
            __m256i sumX2 = _mm256_setzero_si256();
            __m256i sumY2 = _mm256_setzero_si256();

            for (int j = -1; j <= 1; j++) {
                // Calculate offsets
                size_t offset_m1 = (y + j) * width + x - 1;
                size_t offset_0  = (y + j) * width + x;
                size_t offset_p1 = (y + j) * width + x + 1;

                // Load pixels with unaligned memory access
                __m256i row_m1 = _mm256_loadu_si256((__m256i*)&gray_image[offset_m1]);
                __m256i row_0  = _mm256_loadu_si256((__m256i*)&gray_image[offset_0]);
                __m256i row_p1 = _mm256_loadu_si256((__m256i*)&gray_image[offset_p1]);

                // Process lower and higher 128-bit lanes separately
                __m128i row_m1_low = _mm256_extracti128_si256(row_m1, 0);
                __m128i row_m1_high = _mm256_extracti128_si256(row_m1, 1);
                __m128i row_0_low = _mm256_extracti128_si256(row_0, 0);
                __m128i row_0_high = _mm256_extracti128_si256(row_0, 1);
                __m128i row_p1_low = _mm256_extracti128_si256(row_p1, 0);
                __m128i row_p1_high = _mm256_extracti128_si256(row_p1, 1);

                // Convert to 16-bit integers
                __m256i pixels_m1_1 = _mm256_cvtepu8_epi16(row_m1_low);
                __m256i pixels_m1_2 = _mm256_cvtepu8_epi16(row_m1_high);
                __m256i pixels_0_1  = _mm256_cvtepu8_epi16(row_0_low);
                __m256i pixels_0_2  = _mm256_cvtepu8_epi16(row_0_high);
                __m256i pixels_p1_1 = _mm256_cvtepu8_epi16(row_p1_low);
                __m256i pixels_p1_2 = _mm256_cvtepu8_epi16(row_p1_high);

                // Load mask coefficients
                __m256i kx_m1 = _mm256_set1_epi16(Gx[j + 1][0]);
                __m256i kx_0  = _mm256_set1_epi16(Gx[j + 1][1]);
                __m256i kx_p1 = _mm256_set1_epi16(Gx[j + 1][2]);
                __m256i ky_m1 = _mm256_set1_epi16(Gy[j + 1][0]);
                __m256i ky_0  = _mm256_set1_epi16(Gy[j + 1][1]);
                __m256i ky_p1 = _mm256_set1_epi16(Gy[j + 1][2]);

                // Apply masks and accumulate
                // For the lower 16 pixels
                sumX1 = _mm256_add_epi16(sumX1, _mm256_mullo_epi16(kx_m1, pixels_m1_1));
                sumX1 = _mm256_add_epi16(sumX1, _mm256_mullo_epi16(kx_0,  pixels_0_1));
                sumX1 = _mm256_add_epi16(sumX1, _mm256_mullo_epi16(kx_p1, pixels_p1_1));

                sumY1 = _mm256_add_epi16(sumY1, _mm256_mullo_epi16(ky_m1, pixels_m1_1));
                sumY1 = _mm256_add_epi16(sumY1, _mm256_mullo_epi16(ky_0,  pixels_0_1));
                sumY1 = _mm256_add_epi16(sumY1, _mm256_mullo_epi16(ky_p1, pixels_p1_1));

                // For the higher 16 pixels
                sumX2 = _mm256_add_epi16(sumX2, _mm256_mullo_epi16(kx_m1, pixels_m1_2));
                sumX2 = _mm256_add_epi16(sumX2, _mm256_mullo_epi16(kx_0,  pixels_0_2));
                sumX2 = _mm256_add_epi16(sumX2, _mm256_mullo_epi16(kx_p1, pixels_p1_2));

                sumY2 = _mm256_add_epi16(sumY2, _mm256_mullo_epi16(ky_m1, pixels_m1_2));
                sumY2 = _mm256_add_epi16(sumY2, _mm256_mullo_epi16(ky_0,  pixels_0_2));
                sumY2 = _mm256_add_epi16(sumY2, _mm256_mullo_epi16(ky_p1, pixels_p1_2));
            }

            // Approximate magnitude: |sumX| + |sumY|
            __m256i abs_sumX1 = _mm256_abs_epi16(sumX1);
            __m256i abs_sumY1 = _mm256_abs_epi16(sumY1);
            __m256i magnitude1 = _mm256_adds_epu16(abs_sumX1, abs_sumY1);

            __m256i abs_sumX2 = _mm256_abs_epi16(sumX2);
            __m256i abs_sumY2 = _mm256_abs_epi16(sumY2);
            __m256i magnitude2 = _mm256_adds_epu16(abs_sumX2, abs_sumY2);

            // Pack 16-bit results into 8-bit unsigned integers
            __m256i magnitude_packed = _mm256_packus_epi16(magnitude1, magnitude2);

            // Clamp to 8-bit range (though _mm256_packus_epi16 already does this)
            // This step can be redundant but ensures safety
            __m256i magnitude_clamped = _mm256_min_epu8(magnitude_packed, _mm256_set1_epi8(255));

            // Store the result with unaligned memory access
            _mm256_storeu_si256((__m256i*)&output_image[y * width + x], magnitude_clamped);
        }

        // Scalar loop for remaining pixels
        for (; x < width - 1; x++) {
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
            if (magnitude > 255) magnitude = 255;
            output_image[y * width + x] = (unsigned char)magnitude;
        }
    }
}
