#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>

#include "mlp8_model.h"
#include "mlp8_test_vectors.h"


#define NUM_RUNS 1000000


int8_t quantize_int8(float value, float scale)
{
    float q = value / scale;

    int32_t rounded = (int32_t)roundf(q);

    if (rounded > 127)
        rounded = 127;

    if (rounded < -127)
        rounded = -127;

    return (int8_t)rounded;
}


int predict_ecg(const float input[MLP_INPUT_SIZE])
{
    int8_t input_q[MLP_INPUT_SIZE];
    int8_t hidden_q[MLP_HIDDEN_SIZE];
    int32_t acc1[MLP_HIDDEN_SIZE];

    for (int i = 0; i < MLP_INPUT_SIZE; i++)
    {
        float standardized =
            (input[i] - scaler_mean[i])
            / scaler_scale[i];

        input_q[i] =
            quantize_int8(
                standardized,
                input_scale
            );
    }


    for (int j = 0; j < MLP_HIDDEN_SIZE; j++)
    {
        int32_t sum = b1_int32[j];

        for (int i = 0; i < MLP_INPUT_SIZE; i++)
        {
            sum +=
                (int32_t)input_q[i]
                *
                (int32_t)W1_q[
                    i * MLP_HIDDEN_SIZE + j
                ];
        }

        acc1[j] = sum;
    }


    for (int j = 0; j < MLP_HIDDEN_SIZE; j++)
    {
        float hidden_real =
            (float)acc1[j]
            * acc1_scale;

        if (hidden_real < 0.0f)
            hidden_real = 0.0f;

        hidden_q[j] =
            quantize_int8(
                hidden_real,
                hidden_scale
            );
    }


    int32_t acc2 = b2_int32[0];

    for (int j = 0; j < MLP_HIDDEN_SIZE; j++)
    {
        acc2 +=
            (int32_t)hidden_q[j]
            *
            (int32_t)W2_q[j];
    }


    return (acc2 >= 0) ? 1 : 0;
}


int main(void)
{
    const float *test_input =
        test_inputs[0];


    /*
        ----------------------------------------------------
        Verify the model before timing it
        ----------------------------------------------------
    */

    int prediction =
        predict_ecg(test_input);

    int expected =
        expected_predictions[0];


    printf("========================================\n");
    printf("MLP-8 Inference Timing Benchmark\n");
    printf("========================================\n\n");


    printf(
        "Verification prediction: %d\n",
        prediction
    );

    printf(
        "Expected prediction:     %d\n",
        expected
    );


    if (prediction != expected)
    {
        printf("\nERROR: verification failed.\n");
        return 1;
    }


    printf(
        "Verification:             PASS\n\n"
    );


    /*
        ----------------------------------------------------
        Windows high-resolution timer
        ----------------------------------------------------
    */

    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;


    QueryPerformanceFrequency(
        &frequency
    );


    /*
        ----------------------------------------------------
        Warm-up
        ----------------------------------------------------
    */

    for (int i = 0; i < 10000; i++)
    {
        predict_ecg(test_input);
    }


    /*
        ----------------------------------------------------
        Timed benchmark
        ----------------------------------------------------
    */

    QueryPerformanceCounter(
        &start
    );


    volatile int result = 0;


    for (int i = 0; i < NUM_RUNS; i++)
    {
        result ^= predict_ecg(
            test_input
        );
    }


    QueryPerformanceCounter(
        &end
    );


    /*
        ----------------------------------------------------
        Calculate timing
        ----------------------------------------------------
    */

    double elapsed_seconds =
        (double)(end.QuadPart - start.QuadPart)
        /
        (double)frequency.QuadPart;


    double total_ms =
        elapsed_seconds * 1000.0;


    double average_us =
        (elapsed_seconds * 1000000.0)
        /
        NUM_RUNS;


    double inferences_per_second =
        NUM_RUNS
        /
        elapsed_seconds;


    printf(
        "Runs:                     %d\n",
        NUM_RUNS
    );

    printf(
        "Total time:               %.3f ms\n",
        total_ms
    );

    printf(
        "Average inference:        %.3f us\n",
        average_us
    );

    printf(
        "Inferences per second:    %.0f\n",
        inferences_per_second
    );


    /*
        Prevent compiler from completely
        eliminating the benchmark.
    */

    printf(
        "Checksum:                 %d\n",
        result
    );


    printf(
        "\n========================================\n"
    );


    return 0;
}