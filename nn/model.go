package nn

import (
	"encoding/json"
	"errors"
	"os"
)

type ModelData struct {
	Sizes   []int         `json:"sizes"`
	Weights [][][]float64 `json:"weights"`
	Biases  [][]float64   `json:"biases"`
}

// SaveModel сохраняет сеть в JSON-файл
func (net *Network) SaveModel(path string) error {
	model := ModelData{
		Sizes:   net.Sizes(),
		Weights: make([][][]float64, len(net.Layers)),
		Biases:  make([][]float64, len(net.Layers)),
	}

	for i, L := range net.Layers {
		model.Weights[i] = L.Weights
		model.Biases[i] = L.Biases
	}

	data, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// LoadModel загружает сеть из JSON-файла
func LoadModel(path string) (*Network, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var model ModelData
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, err
	}

	if len(model.Sizes) < 2 {
		return nil, errors.New("invalid model format")
	}

	// создаём сеть
	net := NewNetwork(model.Sizes)

	// загружаем веса
	for i := range net.Layers {
		net.Layers[i].Weights = model.Weights[i]
		net.Layers[i].Biases = model.Biases[i]
	}

	return net, nil
}

// Sizes возвращает архитектуру сети
func (net *Network) Sizes() []int {
	s := make([]int, 0, len(net.Layers)+1)
	s = append(s, net.Layers[0].In)
	for _, L := range net.Layers {
		s = append(s, L.Out)
	}
	return s
}
