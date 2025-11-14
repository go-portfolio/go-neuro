package nn

import (
	"encoding/json"
	"os"
)

// -----------------------------
// JSON Model Format
// -----------------------------

// SaveModel сохраняет всю сеть в JSON-файл
func (net *Network) SaveModel(path string) error {
	data, err := json.MarshalIndent(net, "", "  ")
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

	var net Network
	if err := json.Unmarshal(data, &net); err != nil {
		return nil, err
	}

	return &net, nil
}
