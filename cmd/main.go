package main

import (
	"fmt"
	"math"

	"github.com/go-portfolio/go-neuro/nn"
)

func main() {
	// ========================
	// Датасеты по категориям
	// ========================
	datasets := map[string]struct {
		Samples [][]float64
		Targets [][]float64
	}{
		"XOR": {
			Samples: [][]float64{
				{0, 0},
				{0, 1},
				{1, 0},
				{1, 1},
			},
			Targets: [][]float64{
				{0}, {1}, {1}, {0},
			},
		},
		"AND": {
			Samples: [][]float64{
				{0, 0},
				{0, 1},
				{1, 0},
				{1, 1},
			},
			Targets: [][]float64{
				{0}, {0}, {0}, {1},
			},
		},
		"OR": {
			Samples: [][]float64{
				{0, 0},
				{0, 1},
				{1, 0},
				{1, 1},
			},
			Targets: [][]float64{
				{0}, {1}, {1}, {1},
			},
		},
	}

	// ========================
	// Обучение и проверка
	// ========================
	for name, data := range datasets {
		fmt.Printf("\n=== Training %s network ===\n", name)

		// Сеть: 2 входа -> 2 скрытых -> 1 выход
		net := nn.NewNetwork([]int{2, 2, 1})

		// Обучение на всех примерах сразу
		net.TrainBatch(data.Samples, data.Targets, 20000, 0.5)
		fmt.Println("Training done!")

		// Сохранение модели
		modelFile := fmt.Sprintf("%s_model.json", name)
		err := net.SaveModel(modelFile)
		if err != nil {
			panic(err)
		}
		fmt.Printf("Model saved to %s\n", modelFile)

		// Загрузка модели
		loaded, err := nn.LoadModel(modelFile)
		if err != nil {
			panic(err)
		}
		fmt.Println("Model loaded!")

		// Проверка всех примеров
		fmt.Printf("\n=== Predictions for %s ===\n", name)
		for i, s := range data.Samples {
			out := loaded.Predict(s)
			fmt.Printf("Input: %v → Output: %.4f (rounded %d), Target: %v\n",
				s, out[0], int(math.Round(out[0])), data.Targets[i])
		}
	}
}
