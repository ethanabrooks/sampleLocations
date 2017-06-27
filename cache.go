package main

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

type cache struct {
	cost            *mat64.Dense
	choice          *mat64.Dense
	costWithChoices *mat64.Dense
}

// Generates a new `height` x `width` matrix with all values set to NaN
func newCacheMatrix(height int, width int) *mat64.Dense {
	matrix := mat64.NewDense(height, width, nil)
	matrix.Scale(math.NaN(), matrix)
	return matrix
}

func newCache(nChoices int, pathLen int) cache {
	cache := cache{}

	// Store the costs for all intervals [i:j] in path.
	cache.cost = newCacheMatrix(pathLen, pathLen+1)

	// Store the most recent choice for choice number i and path interval starting
	// at j.
	cache.choice = newCacheMatrix(nChoices+1, pathLen+1)

	// Store the cost, taking into account i subsequent choices, for path interval
	// starting at j.
	cache.costWithChoices = newCacheMatrix(nChoices+1, pathLen+1)

	return cache
}

func loadCost(cache *cache, start int, stop int) (float64, bool) {
	cost := cache.cost.At(start, stop)
	return cost, !math.IsNaN(cost)
}

func storeCost(cache *cache, start int, stop int, cost float64) {
	cache.cost.Set(start, stop, cost)
}

func loadBestChoice(cache *cache, nChoices int,
	start int) (int, float64, bool) {
	choice := cache.choice.At(nChoices, start)
	cost := cache.costWithChoices.At(nChoices, start)
	switch {
	case math.IsNaN(choice) && math.IsNaN(cost):
		return 0, 0, false // cache miss
	case !math.IsNaN(choice) && !math.IsNaN(cost):
		return int(choice), cost, true
	default:
		panic("Caches out of sync")
	}
}

func storeBestChoice(cache *cache, nChoices int, start int, choice int, cost float64) {
	cache.choice.Set(nChoices, start, float64(choice))
	cache.costWithChoices.Set(nChoices, start, cost)
}
