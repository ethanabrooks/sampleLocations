package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
	"sync"
	"runtime"
)


func randNormVector(size int) *mat64.Vector {
	vector := mat64.NewVector(size, nil)
	for i := 0; i < size; i++ {
		vector.SetVec(i, rand.NormFloat64()*0.001)
	}
	return vector
}

func setRow(i int, m *mat64.Dense, v *mat64.Vector) {
	_, dim := m.Dims()
	for j := 0; j < dim; j++ {
		m.Set(i, j, v.At(j, 0))
	}
}

func simpleRandomWalk(steps int) *mat64.Dense {
	positions := mat64.NewDense(steps, 1, nil)
	vel := 0
	for i := 1; i < steps; i++ {
		acc := rand.Intn(5)
		if rand.NormFloat64() - .5 > 0 {
			vel += acc
		} else {
			vel -= acc
		}
		oldPos := int(positions.At(i-1, 0))
		positions.Set(i, 0, float64(oldPos+vel))
	}
	return positions
}

func randomWalk(steps int, dim int) *mat64.Dense {
	positions := mat64.NewDense(steps, dim, nil)
	newPos := mat64.NewVector(dim, nil)
	vel := mat64.NewVector(dim, nil)
	for i := 1; i < steps; i++ {
		acc := randNormVector(dim)         // noisey acceleration vector
		vel.AddVec(vel, acc)               // vel += acc
		noise := rand.ExpFloat64() / 0.5   // makes path jagged
		vel.ScaleVec(noise, vel)           // vel *= noise
		oldPos := positions.RowView(i - 1) // oldPos = positions[i - 1]
		newPos.AddVec(oldPos, vel)         // newPos = oldPos + vel
		setRow(i, positions, newPos)       // positions[i] = newPos
	}
	return positions
}

func reverse(numbers []int) []int {
	for i := 0; i < len(numbers)/2; i++ {
		j := len(numbers) - i - 1
		numbers[i], numbers[j] = numbers[j], numbers[i]
	}
	return numbers
}

type CostChoicesPair struct {
	choices []int
	cost    float64
}

type Cache struct {
	getCostLock     sync.Mutex
	bestChoiceLock  sync.Mutex
	cost            *mat64.Dense
	choice          *mat64.Dense
	costWithChoices *mat64.Dense
}

func loadCost(cache *Cache, start int, stop int) (float64, bool) {
	cache.getCostLock.Lock()
	defer cache.getCostLock.Unlock()

	cost := cache.cost.At(start, stop)
	return cost, !math.IsNaN(cost)
}

func storeCost(cache *Cache, start int, stop int, cost float64) {
	cache.getCostLock.Lock()
	defer cache.getCostLock.Unlock()

	cache.cost.Set(start, stop, cost)
}

func loadBestChoice(cache *Cache, nChoices int, start int) (CostChoicesPair, bool) {
	cache.bestChoiceLock.Lock()
	defer cache.bestChoiceLock.Unlock()

	choice := cache.choice.At(nChoices, start)
	cost := cache.costWithChoices.At(nChoices, start)
	switch {
	case math.IsNaN(choice) && math.IsNaN(cost):
		return CostChoicesPair{}, false  // cache miss
	case !math.IsNaN(choice) && !math.IsNaN(cost):

		// walk backward through cache selecting best choices
		var choices []int
		for choiceNum := nChoices - 1; choiceNum > 0; choiceNum-- {
			choice = cache.choice.At(choiceNum, int(choice))
			if !math.IsNaN(choice) {
				choices = append(choices, int(choice))
			}
		}
		return CostChoicesPair{choices, cost}, true
	default:
		panic("Caches out of sync")
	}
}

func storeBestChoice(cache *Cache, nChoices int, start int, value CostChoicesPair) {
	cache.bestChoiceLock.Lock()
	defer cache.bestChoiceLock.Unlock()

	if len(value.choices) > 0 {
		lastChoice := value.choices[len(value.choices) - 1]
		cache.choice.Set(nChoices, start, float64(lastChoice))
	}
	cache.costWithChoices.Set(nChoices, start, value.cost)
}


func getCost(path *mat64.Dense, start int, stop int, cache *Cache) float64 {

	// check if return cost has been cached
	cost, ok := loadCost(cache, start, stop)
	if ok {
		return cost
	}

	// check that stop and start haven't gotten goofed up
	size, dim := path.Dims()
	if start < 0 || stop > size {
		panic(fmt.Sprintf("start (%d) must be positive and stop (%d) must be less "+
			"than path length (%d)", start, stop, size))
	}

	// accumulate cost of path
	cost = 0.0
	for i := start + 1; i < stop; i++ {
		diffsSq := 0.0
		for j := 0; j < dim; j++ {
			diff := path.At(i, j) - path.At(start, j)
			diffsSq += math.Pow(diff, 2)
		}
		cost += math.Sqrt(diffsSq)
	}
	storeCost(cache, start, stop, cost)
	return cost
}

func _bestChoice(nChoices int, path *mat64.Dense, start int, cache *Cache) CostChoicesPair {

	 //check if return value has been cached
	value, ok := loadBestChoice(cache, nChoices, start)
	if ok {
		return value
	}

	stop, _ := path.Dims()

	// running out of choices is terminal condition
	if nChoices == 0 {
		return CostChoicesPair{nil, getCost(path, start, stop, cache)}
	}

	// path is empty is terminal
	if stop - start <= 1 {
		return CostChoicesPair{nil, 0}
	}


	// find the choice of i that minimizes cost
	minCost := math.Inf(1)
	var bestChoices []int
	lock := sync.Mutex{}

	indices := make(chan int)
	go func() {
		for i := start + 1; i < stop; i++ {
			indices <- i
		}
		close(indices)
	}()

	var group sync.WaitGroup
	for i:= 0; i < runtime.NumCPU(); i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			for j := range indices {
				sliceAfter := _bestChoice(nChoices-1, path, j, cache)
				cost := getCost(path, start, j, cache) + sliceAfter.cost
				lock.Lock()
				if cost < minCost {
					minCost = cost
					bestChoices = append(sliceAfter.choices, j)
				}
				lock.Unlock()
			}
		}()
	}
	group.Wait()
	value = CostChoicesPair{bestChoices, minCost}
	storeBestChoice(cache, nChoices, start, value)
	return value
}

func newCacheMatrix(height int, width int) (*mat64.Dense){
	matrix := mat64.NewDense(height, width, nil)
	matrix.Scale(math.NaN(), matrix)
	return matrix
}

func bestChoice(nChoices int, path *mat64.Dense) (float64, []int) {
	pathLen, _ := path.Dims()
	cache := Cache{}

	cache.cost = newCacheMatrix(pathLen, pathLen + 1)
	cache.choice = newCacheMatrix(nChoices + 1, pathLen)
	cache.costWithChoices =	newCacheMatrix(nChoices + 1, pathLen + 1)

	value := _bestChoice(nChoices, path, 0, &cache)
	return value.cost, reverse(value.choices)
}

func main() {
	rand.Seed(3)
	walk := simpleRandomWalk(5)
	//walk := mat64.NewDense(5, 1, []float64{0, 4, 5, 6, 5})
	fmt.Println(mat64.Formatted(walk.T()))
	cost,choices  := bestChoice(2, walk)
	fmt.Println(choices)
	fmt.Println(cost)
}
