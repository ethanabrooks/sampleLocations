package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
	"sync"
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
		if rand.NormFloat64() > 0 {
			vel += acc
		} else {
			vel -= acc
		}
		//vel *= noise           // vel *= noise
		oldPos := int(positions.At(i - 1, 0))
		positions.Set(i, 0, float64(oldPos + vel))
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

func euclidean_distance(a mat64.Matrix, b mat64.Matrix) float64 {
	rowsA, colsA := a.Dims()
	rowsB, colsB := b.Dims()
	if rowsA != rowsB {
		panic(fmt.Sprintf("rows of `a` (%d) not equal to rows of `b` (%d)",
			rowsA, rowsB))
	}
	if colsA != colsB {
		panic(fmt.Sprintf("cols of `a` (%d) not equal to cols of `b` (%d)",
			colsA, colsB))
	}
	sq_distance := 0.0
	for i := 0; i < rowsA; i++ {
		sq_distance += math.Pow(a.At(i, 0)+b.At(i, 0), 2) // (a[i] + b[i])^2
	}
	return math.Sqrt(sq_distance)
}

func get_cost_(path *mat64.Dense) float64 {
	size, _ := path.Dims()
	head := path.RowView(0)
	cost := 0.0
	for i := 0; i < size; i++ {
		pos := path.RowView(i)
		cost += euclidean_distance(head, pos)
	}
	return cost
}



func cost_at(i int, n_choices int,
	path *mat64.Dense, start int, stop int) ([]int, float64){
	choices, cost := bestChoiceSlice(n_choices - 1, path, start + i, stop)
	err := getCost(path, start, start + i)
	new_choices := make([]int, len(choices))
	copy(new_choices, choices)

	for j := range new_choices {
		new_choices[j] += i
	}
	return append(new_choices, i), err + cost
}

func getCost(path *mat64.Dense, start int, stop int) float64 {
	size, dim := path.Dims()
	if start < 0 || stop > size {
		panic(fmt.Sprintf( "start (%d) must be positive and stop (%d) must be less " +
			"than path length (%d)", start, stop, size))
	}

	sq_cost := 0.0

	//TODO: parallel
	for i := start + 1; i < stop; i ++ {
		for j := 0; j < dim; j++ {
			diff := path.At(i, j) - path.At(start, j)
			sq_cost += math.Pow(diff, 2)
		}
	}
	return math.Sqrt(sq_cost)
}

func hashCode(path *mat64.Dense, start int, stop int, nChoices ...int) string {
	return fmt.Sprint(nChoices, mat64.Formatted(path.T()), start, stop)
}

type costChoices struct {
	cost float64
	choices []int
}

type Cache struct{
	cache map[string]struct{
		cost float64
		choices []int
	}
	lock sync.RWMutex
}


func bestChoiceSlice(nChoices int, path *mat64.Dense, start int, stop int) ([]int, float64) {
	if nChoices == 0 {
		//key := hashCode(path, start, stop)
		//var cost float64
		//if value, ok := cache.cache[key]; ok {
		//	cost = value.cost
		//} else {
		//	cost = getCost(path, start, stop)
		//}
		return make([]int, 0, 50), getCost(path, start, stop)
	} else {
		minCost := math.Inf(1)
		var bestChoices []int

		for i := start + 1; i < stop; i++ {
			err := getCost(path, start, i)
			choices, cost := bestChoiceSlice(nChoices - 1, path, i, stop)
			if err + cost < minCost {
				minCost = err + cost
				bestChoices = append(choices, i)
			}
		}
		return bestChoices, minCost
	}
}

func bestChoice(nChoices int, path *mat64.Dense) ([]int, float64) {
	size, _ := path.Dims()
	//cache := Cache{}
	return bestChoiceSlice(nChoices, path, 0, size)
}

func main() {
	rand.Seed(2)
	walk := simpleRandomWalk(20)
	choices, cost := bestChoice(5, walk)
	fmt.Println(choices)
	fmt.Println(cost)
}
