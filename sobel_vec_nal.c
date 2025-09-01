// sobel_dll.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

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
        // Обработка по 16 пикселей за раз
        for (; x <= width - 1 - 16; x += 16) {
            // Инициализация сумм
            __m256i sumX = _mm256_setzero_si256();
            __m256i sumY = _mm256_setzero_si256();

            for (int j = -1; j <= 1; j++) {
                // Загрузка 16 пикселей
                __m128i row_pixels = _mm_loadu_si128((__m128i*)&gray_image[(y + j) * width + (x - 1)]);

                // Преобразование в 16-битные целые числа
                __m256i pixels = _mm256_cvtepu8_epi16(row_pixels);

                // Загружаем смещённые пиксели
                __m128i row_pixels_shifted_left = _mm_loadu_si128((__m128i*)&gray_image[(y + j) * width + x]);
                __m256i pixels_shifted_left = _mm256_cvtepu8_epi16(row_pixels_shifted_left);

                __m128i row_pixels_shifted_right = _mm_loadu_si128((__m128i*)&gray_image[(y + j) * width + (x + 1)]);
                __m256i pixels_shifted_right = _mm256_cvtepu8_epi16(row_pixels_shifted_right);

                // Коэффициенты масок
                __m256i kx_left = _mm256_set1_epi16(Gx[j + 1][0]);
                __m256i ky_left = _mm256_set1_epi16(Gy[j + 1][0]);

                __m256i kx_center = _mm256_set1_epi16(Gx[j + 1][1]);
                __m256i ky_center = _mm256_set1_epi16(Gy[j + 1][1]);

                __m256i kx_right = _mm256_set1_epi16(Gx[j + 1][2]);
                __m256i ky_right = _mm256_set1_epi16(Gy[j + 1][2]);

                // Вычисление сумм
                sumX = _mm256_add_epi16(sumX, _mm256_mullo_epi16(kx_left, pixels));
                sumY = _mm256_add_epi16(sumY, _mm256_mullo_epi16(ky_left, pixels));

                sumX = _mm256_add_epi16(sumX, _mm256_mullo_epi16(kx_center, pixels_shifted_left));
                sumY = _mm256_add_epi16(sumY, _mm256_mullo_epi16(ky_center, pixels_shifted_left));

                sumX = _mm256_add_epi16(sumX, _mm256_mullo_epi16(kx_right, pixels_shifted_right));
                sumY = _mm256_add_epi16(sumY, _mm256_mullo_epi16(ky_right, pixels_shifted_right));
            }

            // Преобразование в 32-битные целые для предотвращения переполнения
            __m256i sumX_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sumX, 0));
            __m256i sumX_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sumX, 1));

            __m256i sumY_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sumY, 0));
            __m256i sumY_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sumY, 1));

            // Преобразование в плавающую точку
            __m256 sumX_lo_ps = _mm256_cvtepi32_ps(sumX_lo);
            __m256 sumX_hi_ps = _mm256_cvtepi32_ps(sumX_hi);

            __m256 sumY_lo_ps = _mm256_cvtepi32_ps(sumY_lo);
            __m256 sumY_hi_ps = _mm256_cvtepi32_ps(sumY_hi);

            // Вычисление magnitude
            __m256 magnitude_lo_ps = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(sumX_lo_ps, sumX_lo_ps), _mm256_mul_ps(sumY_lo_ps, sumY_lo_ps)));
            __m256 magnitude_hi_ps = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(sumX_hi_ps, sumX_hi_ps), _mm256_mul_ps(sumY_hi_ps, sumY_hi_ps)));

            // Преобразование обратно в 32-битные целые
            __m256i magnitude_lo_epi32 = _mm256_cvtps_epi32(magnitude_lo_ps);
            __m256i magnitude_hi_epi32 = _mm256_cvtps_epi32(magnitude_hi_ps);

            // Ограничение значений от 0 до 255
            __m256i magnitude_lo_epi16 = _mm256_packs_epi32(magnitude_lo_epi32, magnitude_hi_epi32);
            magnitude_lo_epi16 = _mm256_min_epi16(magnitude_lo_epi16, _mm256_set1_epi16(255));
            magnitude_lo_epi16 = _mm256_max_epi16(magnitude_lo_epi16, _mm256_setzero_si256());

            // Преобразование в 8-битные целые
            __m128i result = _mm_packus_epi16(_mm256_extracti128_si256(magnitude_lo_epi16, 0), _mm256_extracti128_si256(magnitude_lo_epi16, 1));

            // Сохранение результатов
            _mm_storeu_si128((__m128i*)&output_image[y * width + x], result);
        }

        // Обработка оставшихся пикселей
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

            int magnitude = (int)sqrt((double)(sumX * sumX + sumY * sumY));
            if (magnitude > 255) magnitude = 255;
            if (magnitude < 0) magnitude = 0;
            output_image[y * width + x] = (unsigned char)magnitude;
        }
    }
}
