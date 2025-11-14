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
		Index   int // индекс выхода для этой задачи
	}{
		"XOR": {
			Samples: [][]float64{
				{0, 0}, {0, 1}, {1, 0}, {1, 1},
			},
			Targets: [][]float64{
				{1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0},
			},
			Index: 0,
		},
		"AND": {
			Samples: [][]float64{
				{0, 0}, {0, 1}, {1, 0}, {1, 1},
			},
			Targets: [][]float64{
				{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
			},
			Index: 1,
		},
		"OR": {
			Samples: [][]float64{
				{0, 0}, {0, 1}, {1, 0}, {1, 1},
			},
			Targets: [][]float64{
				{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
			},
			Index: 2,
		},
	}

	// ========================
	// Объединяем все данные
	// ========================
	allSamples := [][]float64{}
	allTargets := [][]float64{}
	for _, data := range datasets {
		allSamples = append(allSamples, data.Samples...)
		allTargets = append(allTargets, data.Targets...)
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
	net.TrainBatch(allSamples, allTargets, epochs, lr)
	fmt.Println("Training done!")

	// ========================
	// Сохранение модели
	// ========================
	modelFile := "all_tasks_model.json"
	if err := net.SaveModel(modelFile); err != nil {
		panic(err)
	}
	fmt.Println("Model saved!")

	// ========================
	// Загрузка модели
	// ========================
	loaded, err := nn.LoadModel(modelFile)
	if err != nil {
		panic(err)
	}
	fmt.Println("Model loaded!")

	// ========================
	// Проверка всех задач
	// ========================
	for name, data := range datasets {
		fmt.Printf("\n=== Predictions for %s ===\n", name)
		for i, s := range data.Samples {
			out := loaded.Predict(s)
			// выбираем соответствующий выход
			val := out[data.Index]
			fmt.Printf("Input: %v → Output: %.4f (rounded %d), Target: %.0f\n",
				s, val, int(math.Round(val)), data.Targets[i][data.Index])
		}
	}
}
