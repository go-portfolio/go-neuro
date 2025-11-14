package main

import (
	"fmt"
	"math"
	"os"

	"github.com/go-portfolio/go-neuro/nn"
)

func main() {
	// Создаём/пересоздаём общий лог
	f, err := os.Create("full_log.txt")
	if err != nil {
		fmt.Println("Ошибка создания файла:", err)
		return
	}
	defer f.Close()

	// ========================
	// Входные данные
	// ========================
	samples := [][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}

	targets := [][]float64{
		{0, 0, 0}, {1, 0, 1}, {1, 0, 1}, {0, 1, 1},
	}

	// ========================
	// Создаем сеть
	// ========================
	net := nn.NewNetwork([]int{2, 10, 3})

	fmt.Fprintln(f, "=== Training Start ===")
	// ========================
	// Обучение с логом в файл
	// ========================
	// SGD с мини-батчами
	net.TrainMiniBatchSGD(samples, targets, 20000, 0.5, 2, f)

	fmt.Fprintln(f, "Training done!\n")

	// ========================
	// Проверка сети и вывод активаций на каждом слое
	// ========================
	for si, s := range samples {
		fmt.Fprintf(f, "\n=== Sample %d: Input=%v ===\n", si, s)
		out, activations, zvals := net.ForwardFull(s, false)


		for li := 0; li < len(activations); li++ {
			if li == 0 {
				fmt.Fprintf(f, "\n--- Layer %d (Input Layer, neurons=%d) ---\n", li, len(activations[li]))
			} else {
				layer := net.Layers[li-1]
				fmt.Fprintf(f, "\n--- Layer %d (neurons=%d) ---\n", li, layer.Out)
				for ni := 0; ni < layer.Out; ni++ {
					fmt.Fprintf(f, "Neuron %d: Weights=%v, Bias=%.4f, Z=%.4f → A=%.4f\n",
						ni, layer.Weights[ni], layer.Biases[ni], zvals[li-1][ni], activations[li][ni])
				}
			}
			fmt.Fprintf(f, "Layer %d outputs (A): %v\n", li, activations[li])
		}

		fmt.Fprintf(f, "\nNetwork Output:\n")
		for ni, v := range out {
			fmt.Fprintf(f, " Output neuron %d: %.4f (rounded %d), Target: %.0f\n",
				ni, v, int(math.Round(v)), targets[si][ni])
		}
	}

	// ========================
	// Предсказания по задачам
	// ========================
	tasks := map[string]int{"XOR": 0, "AND": 1, "OR": 2}
	for name, idx := range tasks {
		fmt.Fprintf(f, "\n=== Predictions for %s ===\n", name)
		for i, s := range samples {
			out := net.Predict(s)
			fmt.Fprintf(f, "Input: %v → Output: %.4f (rounded %d), Target: %.0f\n",
				s, out[idx], int(math.Round(out[idx])), targets[i][idx])
		}
	}
	fmt.Printf("Модель создана успешно. Логи выложены в файл `full_log.txt`")
}
