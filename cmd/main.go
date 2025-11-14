package main

import (
	"fmt"
	"math"
	"github.com/go-portfolio/go-neuro/nn"
	"github.com/go-portfolio/go-neuro/nn/dataset"
	"github.com/go-portfolio/go-neuro/nn/trainers"
)

func main() {
	// 1. Берём датасет
	ds := dataset.MustGet("xor")

	// 2. Создаём сеть
	net := nn.NewNetwork([]int{2, 2, 1})

	// 3. Выбираем тренер
	trainer := &trainers.SGDTrainer{LearningRate: 0.5}

	// 4. Обучаем сеть
	net.Fit(trainer, ds.Samples, ds.Targets, 20000)

	fmt.Println("\nTraining done!")

	// 5. Сохраняем модель
	err := net.SaveModel("xor_model.json")
	if err != nil {
		panic(err)
	}
	fmt.Println("Model saved → xor_model.json")

	// 6. Загружаем модель
	loadedNet, err := nn.LoadModel("xor_model.json")
	if err != nil {
		panic(err)
	}
	fmt.Println("Model loaded!")

	// 7. Проверяем предсказания загруженной сети
	fmt.Println("\n--- Prediction check ---")
	for i, s := range ds.Samples {
		out := loadedNet.Predict(s)
		fmt.Printf(
			"Input: %v → Output: %.4f (rounded %d), Target: %v\n",
			s,
			out[0],
			int(math.Round(out[0])),
			ds.Targets[i],
		)
	}
}
