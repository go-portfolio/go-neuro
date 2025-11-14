package main

import (
	"fmt"
	"math"

	"github.com/go-portfolio/go-neuro/nn"
)

func main() {
	// ========================
	// Входные данные
	// ========================
	samples := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	// ========================
	// Целевые значения для XOR, AND, OR
	// neuron0 → XOR, neuron1 → AND, neuron2 → OR
	// ========================
	targets := [][]float64{
		{0 ^ 0, 0 & 0, 0 | 0},
		{0 ^ 1, 0 & 1, 0 | 1},
		{1 ^ 0, 1 & 0, 1 | 0},
		{1 ^ 1, 1 & 1, 1 | 1},
	}

	// ========================
	// Создаем сеть: 2 → 6 → 3
	// ========================
	net := nn.NewNetwork([]int{2, 6, 3})

	// ========================
	// Обучение
	// ========================
	epochs := 20000
	lr := 0.5
	net.TrainBatch(samples, targets, epochs, lr)
	fmt.Println("Training done!")

	// ========================
	// Проверка сети и вывод активаций на каждом слое
	// ========================
	for si, s := range samples {
		fmt.Printf("\n=== Sample %d: Input=%v ===\n", si, s)

		// Forward pass с сохранением активаций и Z
		_, activations, zvals := net.ForwardFull(s)

		// Вывод активаций на каждом слое, включая входной
		for li := 0; li < len(activations); li++ {
			if li == 0 {
				fmt.Printf("\n--- Layer %d (Input Layer, neurons=%d) ---\n", li, len(activations[li]))
			} else {
				layer := net.Layers[li-1]
				fmt.Printf("\n--- Layer %d (neurons=%d) ---\n", li, layer.Out)
				fmt.Printf("Inputs: %d\n", layer.In)
				for ni := 0; ni < layer.Out; ni++ {
					fmt.Printf("Neuron %d: Weights=%v, Bias=%.4f\n", ni, layer.Weights[ni], layer.Biases[ni])
					fmt.Printf("  Z=%.4f → A=%.4f\n", zvals[li-1][ni], activations[li][ni])
				}
			}
			// Вывод выходных данных слоя
			fmt.Printf("Layer %d outputs (A): %v\n", li, activations[li])
		}

		// Печать итогового выхода сети
		fmt.Printf("\n=== Network Output ===\n")
		for ni, v := range activations[len(activations)-1] {
			fmt.Printf("Output neuron %d: %.4f (rounded %d), Target: %.0f\n",
				ni, v, int(math.Round(v)), targets[si][ni])
		}
	}

	// ========================
	// Вывод предсказаний по задачам
	// ========================
	tasks := map[string]int{
		"XOR": 0,
		"AND": 1,
		"OR":  2,
	}

	for name, idx := range tasks {
		fmt.Printf("\n=== Predictions for %s ===\n", name)
		for i, s := range samples {
			out := net.Predict(s)
			fmt.Printf("Input: %v → Output: %.4f (rounded %d), Target: %.0f\n",
				s, out[idx], int(math.Round(out[idx])), targets[i][idx])
		}
	}
}
