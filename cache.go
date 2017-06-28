package main

import (
	"github.com/gonum/matrix/mat64"
	"math"
	"log"
)

// Cache is a struct of matrices used for caching by the functions in sampleLocations.go
type cache struct {
	cost            *mat64.Dense
	choice          *mat64.Dense
	costWithChoices *mat64.Dense
}

// NewCacheMatrix generates a new `height` x `width` matrix with all values set to NaN
func newCacheMatrix(height, width int) *mat64.Dense {
	matrix := mat64.NewDense(height, width, nil)
	matrix.Scale(math.NaN(), matrix)
	return matrix
}

// NewCache creates a `nChoices` x `pathLen` size matrix for use in the cache struct
func newCache(nChoices, pathLen int) cache {
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

// LoadCost checks if the cache contains the output of `getCost(path, start, stop, cache)`
// and returns `(output, ok)` where `ok` is true iff the cache contains the value
func loadCost(cache *cache, start, stop int) (float64, bool) {
	cost := cache.cost.At(start, stop)
	return cost, !math.IsNaN(cost)
}

// StoreCost caches the output of `getCost(path, start, stop, cache)`
func storeCost(cache *cache, start, stop int, cost float64) {
	cache.cost.Set(start, stop, cost)
}

// LoadNextChoice checks if the cache contains the output of `nextChoice(nChoices, path, start, cache)`
// and returns `(output, ok)` where `ok` is true iff the cache contains the value
func loadNextChoice(cache *cache, nChoices, start int) (int, float64, bool) {
	choice := cache.choice.At(nChoices, start)
	cost := cache.costWithChoices.At(nChoices, start)
	switch {
	case math.IsNaN(choice) && math.IsNaN(cost):
		return 0, 0, false // cache miss
	case !math.IsNaN(choice) && !math.IsNaN(cost):
		return int(choice), cost, true
	default:
		log.Fatal("Caches out of sync.")
		return 0, 0, false
	}
}

// StoreNextChoice caches the output of `nextChoice(nChoices, path, start, cache)`
func storeNextChoice(cache *cache, nChoices, start, choice int, cost float64) {
	cache.choice.Set(nChoices, start, float64(choice))
	cache.costWithChoices.Set(nChoices, start, cost)
}
