#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

DLL_EXPORT void apply_sobel(unsigned char *gray_image, int width, int height, unsigned char *output_image) {
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
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

    // Применение оператора Собеля к каждому пикселю, исключая границы
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumX = 0;
            int sumY = 0;

            // Применение масок Gx и Gy
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    int pixel = gray_image[(y + j) * width + (x + i)];
                    sumX += Gx[j + 1][i + 1] * pixel;
                    sumY += Gy[j + 1][i + 1] * pixel;
                }
            }

            // Вычисление градиента
            int magnitude = (int)sqrt((double)(sumX * sumX + sumY * sumY));

            // Ограничение значения до 255
            if (magnitude > 255) magnitude = 255;
            if (magnitude < 0) magnitude = 0;

            output_image[y * width + x] = (unsigned char)magnitude;
        }
    }
}
