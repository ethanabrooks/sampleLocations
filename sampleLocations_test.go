package main

import (
	"github.com/gonum/matrix/mat64"
	"math/rand"
	"testing"
	"fmt"
)

func ExampleBestChoices() {
	rand.Seed(0)
	walk := simpleRandomWalk(6)
	// Output: [ 0   4  11  18  23  31]
	fmt.Println(mat64.Formatted(walk.T()))
	cost, choices := BestChoices(3, walk)
	fmt.Println(choices)
	// Output: [2 3 5]
	fmt.Println(cost)
	// Output: 9

}

// GetCostNoCache calculates the sum of mean squared errors from the first point in the path.
func getCostNoCache(path *mat64.Dense, start, stop int) float64 {
	cost := 0.0
	for i := start; i < stop; i++ {
		cost += euclideanDistance(path, i, i+1)
	}
	return cost
}

// CostWithChoices calculates the cost (a la `getCost`) for each segment of the path
// delineated by `choices`. If choices are [1, 3], this function will calculate
// cost(path[:1]) + cost(path[1:3]) + cost(path[3:])
func costWithChoices(path *mat64.Dense, choices []int) float64 {
	size, _ := path.Dims()
	choices = append(choices, size)
	cost := .0
	prevChoice := 0
	for _, choice := range choices {
		cost += getCostNoCache(path, prevChoice, choice-1)
		prevChoice = choice
	}
	return cost
}

// TestCostWithChoices checks that costWithChoices gives the same results as
// getCost both with 0 choices and with 1 choice.
func TestCostWithChoices(t *testing.T) {
	length := rand.Intn(9) + 1
	path := simpleRandomWalk(length)

	// Compare costWithChoices to original getCost function with 0 choices.
	costWithChoicesValue := costWithChoices(path, []int{})
	cache := newCache(1, length)
	getCostValue := getCost(path, 0, length, &cache)
	if costWithChoicesValue != getCostValue {
		t.Errorf("These values should be equal:\n"+
			"Result from costWithChoices: %f\n"+
			"Result from getCost: %v\n"+
			"Path: %v.",
			costWithChoicesValue, getCostValue, mat64.Formatted(path.T()))
	}

	// Compare costWithChoices to original getCost function with 1 random choice.
	randomChoice := rand.Intn(length)
	costWithChoicesValue = costWithChoices(path, []int{randomChoice})
	cost1 := getCost(path, 0, randomChoice, &cache)
	cost2 := getCost(path, randomChoice, length, &cache)
	getCostValue = cost1 + cost2
	if costWithChoicesValue != getCostValue {
		t.Errorf("These values should be equal:\n"+
			"Result from costWithChoices: %f\n"+
			"Result from getCost: %v\n"+
			"Choice: %d\n"+
			"Path: %v",
			costWithChoicesValue, getCostValue, randomChoice, mat64.Formatted(path.T()))
	}
}

// TestBestChoices ensures that BestChoices is never outperformed (in terms of cost)
// by randomly chosen choices.
func TestBestChoices(t *testing.T) {
	length := rand.Intn(9) + 1
	path := simpleRandomWalk(length)
	nChoices := rand.Intn(length-2) + 1
	for i := 0; i < 20; i++ {
		rand.Seed(int64(i))
		randomChoices := make([]int, nChoices)
		for range randomChoices {
			randomChoices = append(randomChoices, rand.Intn(length))
		}
		algorithmChoices, algorithmCost := BestChoices(nChoices, path)

		// Ensure that algorithm reports accurate costs.
		actualCost := costWithChoices(path, algorithmChoices)
		if algorithmCost != actualCost {
			t.Errorf("These values should be equal:\n"+
				"Result from BestChoices: %f\n"+
				"Result from costWithChoices: %f\n"+
				"Choices: %v\n"+
				"Path: %v",
				algorithmCost, actualCost, algorithmChoices, path)
		}

		// Ensure that cost of algorithm choices can't be beat by random choices.
		randomCost := costWithChoices(path, randomChoices)
		if algorithmCost > randomCost {
			t.Errorf("Cost for choices from BestChoices should be lower than cost for random choices.\n"+
				"Choices from BestChoices: %v\n"+
				"Cost for BestChoices: %f\n"+
				"Random choices: %v\n"+
				"Cost for random choices: %f\n"+
				"Path: %v",
				algorithmChoices, algorithmCost, randomChoices, randomCost, path)
		}
	}
}
