package main

import (
	"testing"
	"math/rand"
	"github.com/gonum/matrix/mat64"
	"fmt"
)

func getCostNoCache(path *mat64.Dense, start, stop int) (float64) {
	cost := 0.0
	for i:=start; i < stop; i++ {
		cost += euclideanDistance(path, i, i+1)
	}
	return cost
}

func costWithChoices(path *mat64.Dense, choices []int) (float64) {
	size, _ := path.Dims()
	choices = append(choices, size)
	cost := .0
	prevChoice := 0
	for _, choice := range choices {
		cost += getCostNoCache(path, prevChoice, choice - 1)
		prevChoice = choice
	}
	return cost
}

func TestCostWithChoices(*testing.T) {
	length := rand.Intn(9) + 1
	path := simpleRandomWalk(length)

	// Compare costWithChoices to original getCost function with 0 choices.
	costWithChoicesValue := costWithChoices(path, []int{})
	cache := newCache(1, length)
	getCostValue := getCost(path, 0, length, &cache)
	if costWithChoicesValue != getCostValue {
		panic(fmt.Sprintf("difference between costWithChoices and getCost: %f vs %f" +
			"\nPath: %v",
			costWithChoicesValue, getCostValue, mat64.Formatted(path.T())))
	}

	// Compare costWithChoices to original getCost function with 1 random choice.
	randomChoice := rand.Intn(length)
	costWithChoicesValue = costWithChoices(path, []int{randomChoice})
	cost1 := getCost(path, 0, randomChoice, &cache)
	cost2 := getCost(path, randomChoice, length, &cache)
	getCostValue = cost1 + cost2
	if costWithChoicesValue != getCostValue {
		panic(fmt.Sprintf("costWithChoices: %f\ngetCost: %f",
			costWithChoicesValue, getCostValue))
	}
}

func TestBestChoices(*testing.T) {
	length := rand.Intn(9) + 1
	path := simpleRandomWalk(length)
	nChoices := rand.Intn(length - 2) + 1
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
			panic(fmt.Sprintf("algorithmCost: %f\nactualCost: %f",
				algorithmCost, actualCost))
		}

		// Ensure that cost of algorithm choices can't be beat by random choices.
		randomCost := costWithChoices(path, randomChoices)
		if algorithmCost > randomCost {
			panic(fmt.Sprintf("algorithm had a higher cost than random:" +
				"algorithm cost: %f\nrandom cost:%f", algorithmCost, actualCost))
		}
	}
}

