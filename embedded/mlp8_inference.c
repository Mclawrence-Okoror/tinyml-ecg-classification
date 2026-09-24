#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "mlp8_model.h"
#include "mlp8_test_vectors.h"


/*
    MLP-8 integer-style ECG classifier

    Python reference:

        ECG
          ↓
        StandardScaler
          ↓
        int8 input
          ↓
        int8 × int8 → int32
          ↓
        ReLU
          ↓
        int8 hidden layer
          ↓
        int8 × int8 → int32
          ↓
        binary prediction
*/


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
        ============================================
        1. Standardize and quantize ECG input
        ============================================
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
        ============================================
        2. First neural-network layer
        ============================================

        int8 × int8 → int32
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
        ============================================
        3. ReLU + hidden quantization
        ============================================
    */

    for (int j = 0; j < MLP_HIDDEN_SIZE; j++)
    {
        float hidden_real =
            (float)acc1[j]
            * acc1_scale;


        /* ReLU */

        if (hidden_real < 0.0f)
            hidden_real = 0.0f;


        hidden_q[j] =
            quantize_int8(
                hidden_real,
                hidden_scale
            );
    }


    /*
        ============================================
        4. Second neural-network layer
        ============================================
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
        ============================================
        5. Binary classification
        ============================================

        sigmoid(logit) >= 0.5

        is mathematically equivalent to:

        logit >= 0

        Because acc2_scale is positive,
        acc2 >= 0 gives the same decision.
    */

    if (acc2 >= 0)
        return 1;

    return 0;
}


int main(void)
{
    int correct = 0;
    int mismatches = 0;


    printf("========================================\n");
    printf("MLP-8 Python -> C parity test\n");
    printf("========================================\n\n");

    printf(
        "Testing %d ECG beats...\n\n",
        NUM_TEST_VECTORS
    );


    /*
        Run all 100 test beats
    */

    for (int i = 0; i < NUM_TEST_VECTORS; i++)
    {
        int c_prediction =
            predict_ecg(test_inputs[i]);

        int python_prediction =
            expected_predictions[i];

        int true_label =
            test_labels[i];


        if (c_prediction == python_prediction)
        {
            correct++;
        }
        else
        {
            mismatches++;

            printf(
                "MISMATCH #%d\n"
                "  Vector: %d\n"
                "  True label: %d\n"
                "  Python: %d\n"
                "  C: %d\n\n",

                mismatches,
                i,
                true_label,
                python_prediction,
                c_prediction
            );
        }
    }


    printf("----------------------------------------\n");

    printf(
        "Python/C agreement: %d / %d\n",
        correct,
        NUM_TEST_VECTORS
    );

    printf(
        "Agreement: %.2f%%\n",
        100.0 * correct / NUM_TEST_VECTORS
    );

    printf(
        "Mismatches: %d\n",
        mismatches
    );

    printf("----------------------------------------\n");


    if (mismatches == 0)
    {
        printf(
            "\nSUCCESS: C exactly matches "
            "the Python integer reference.\n"
        );
    }
    else
    {
        printf(
            "\nC and Python do not yet match.\n"
        );
    }


    return 0;
}