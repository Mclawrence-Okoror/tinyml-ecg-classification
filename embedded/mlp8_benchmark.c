#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "mlp8_model.h"
#include "mlp8_test_vectors.h"


/*
    ============================================================
    MLP-8 Embedded Benchmark
    ============================================================

    Measures:

    1. Model constant storage
    2. Working memory used by inference
    3. Approximate operation count
*/


#define INPUT_BYTES \
    (MLP_INPUT_SIZE * sizeof(int8_t))

#define HIDDEN_BYTES \
    (MLP_HIDDEN_SIZE * sizeof(int8_t))

#define ACC1_BYTES \
    (MLP_HIDDEN_SIZE * sizeof(int32_t))


/*
    Temporary memory used during inference.

    input_q
    hidden_q
    acc1
*/

#define INFERENCE_RAM \
    (INPUT_BYTES + HIDDEN_BYTES + ACC1_BYTES)


/*
    Number of MAC operations.

    Layer 1:
        90 inputs × 8 neurons

    Layer 2:
        8 inputs × 1 output
*/

#define LAYER1_MACS \
    (MLP_INPUT_SIZE * MLP_HIDDEN_SIZE)

#define LAYER2_MACS \
    (MLP_HIDDEN_SIZE)

#define TOTAL_MACS \
    (LAYER1_MACS + LAYER2_MACS)


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


    /*
        1. Standardize + quantize input
    */

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


    /*
        2. First layer
    */

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


    /*
        3. ReLU + hidden quantization
    */

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


    /*
        4. Second layer
    */

    int32_t acc2 = b2_int32[0];

    for (int j = 0; j < MLP_HIDDEN_SIZE; j++)
    {
        acc2 +=
            (int32_t)hidden_q[j]
            *
            (int32_t)W2_q[j];
    }


    /*
        5. Classification
    */

    return (acc2 >= 0) ? 1 : 0;
}


int main(void)
{
    printf("========================================\n");
    printf("MLP-8 Embedded Cost Benchmark\n");
    printf("========================================\n\n");


    printf("MODEL\n");
    printf("----------------------------------------\n");

    printf(
        "Input samples:       %d\n",
        MLP_INPUT_SIZE
    );

    printf(
        "Hidden neurons:      %d\n",
        MLP_HIDDEN_SIZE
    );

    printf(
        "Parameters:          737\n"
    );

    printf(
        "INT8 weights:        %zu bytes\n",
        sizeof(W1_q) + sizeof(W2_q)
    );

    printf(
        "INT32 biases:        %zu bytes\n",
        sizeof(b1_int32) + sizeof(b2_int32)
    );

    printf(
        "Scaler mean:         %zu bytes\n",
        sizeof(scaler_mean)
    );

    printf(
        "Scaler scale:        %zu bytes\n",
        sizeof(scaler_scale)
    );

    printf(
        "Quantization scales: %zu bytes\n",
        sizeof(input_scale)
        + sizeof(W1_scale)
        + sizeof(hidden_scale)
        + sizeof(W2_scale)
    );


    printf("\n");


    printf("INFERENCE RAM\n");
    printf("----------------------------------------\n");

    printf(
        "Input int8 buffer:   %zu bytes\n",
        (size_t)INPUT_BYTES
    );

    printf(
        "Hidden int8 buffer:  %zu bytes\n",
        (size_t)HIDDEN_BYTES
    );

    printf(
        "Accumulator buffer:  %zu bytes\n",
        (size_t)ACC1_BYTES
    );

    printf(
        "Total inference RAM: %zu bytes\n",
        (size_t)INFERENCE_RAM
    );


    printf("\n");


    printf("COMPUTATION\n");
    printf("----------------------------------------\n");

    printf(
        "Layer 1 MACs:        %d\n",
        LAYER1_MACS
    );

    printf(
        "Layer 2 MACs:        %d\n",
        LAYER2_MACS
    );

    printf(
        "Total MACs:          %d\n",
        TOTAL_MACS
    );


    printf("\n");


    /*
        Run one real ECG beat to make sure
        the benchmark is using the actual model.
    */

    int prediction =
        predict_ecg(test_inputs[0]);


    printf("TEST INFERENCE\n");
    printf("----------------------------------------\n");

    printf(
        "Prediction:          %d\n",
        prediction
    );

    printf(
        "Expected:            %d\n",
        expected_predictions[0]
    );


    if (prediction == expected_predictions[0])
    {
        printf(
            "Verification:        PASS\n"
        );
    }
    else
    {
        printf(
            "Verification:        FAIL\n"
        );
    }


    printf("\n");
    printf("========================================\n");


    return 0;
}