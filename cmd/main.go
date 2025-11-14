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
	// Объединяем все данные для обучения одной сети
	// ========================
	allSamples := [][]float64{}
	allTargets := [][]float64{}

	for _, data := range datasets {
		allSamples = append(allSamples, data.Samples...)
		allTargets = append(allTargets, data.Targets...)
	}

	// ========================
	// Создаем сеть
	// 2 входа -> 4 скрытых -> 1 выход
	// Скрытый слой больше, чтобы сеть могла справиться с разными задачами
	// ========================
	net := nn.NewNetwork([]int{2, 4, 1})

	// Обучение
	epochs := 20000
	lr := 0.5
	net.TrainBatch(allSamples, allTargets, epochs, lr)
	fmt.Println("Training done!")

	// Сохранение модели
	modelFile := "all_tasks_model.json"
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

	// ========================
	// Проверка всех задач
	// ========================
	for name, data := range datasets {
		fmt.Printf("\n=== Predictions for %s ===\n", name)
		for i, s := range data.Samples {
			out := loaded.Predict(s)
			fmt.Printf("Input: %v → Output: %.4f (rounded %d), Target: %v\n",
				s, out[0], int(math.Round(out[0])), data.Targets[i])
		}
	}
}
