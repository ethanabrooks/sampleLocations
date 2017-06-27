package main

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

type Cache struct {
	cost            *mat64.Dense
	choice          *mat64.Dense
	costWithChoices *mat64.Dense
}

func loadCost(cache *Cache, start int, stop int) (float64, bool) {
	cost := cache.cost.At(start, stop)
	return cost, !math.IsNaN(cost)
}

func storeCost(cache *Cache, start int, stop int, cost float64) {
	cache.cost.Set(start, stop, cost)
}

func loadBestChoice(cache *Cache, nChoices int,
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

func storeBestChoice(cache *Cache, nChoices int, start int, choice int, cost float64) {
	cache.choice.Set(nChoices, start, float64(choice))
	cache.costWithChoices.Set(nChoices, start, cost)
}
