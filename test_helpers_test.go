package inference

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

func checkNoError(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func checkError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func checkEqual(t *testing.T, want, got any) {
	t.Helper()
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("want %v, got %v", want, got)
	}
}

func checkNotEqual(t *testing.T, want, got any) {
	t.Helper()
	if reflect.DeepEqual(want, got) {
		t.Fatalf("expected values to differ, got %v", got)
	}
}

func checkTrue(t *testing.T, cond bool) {
	t.Helper()
	if !cond {
		t.Fatal("expected true")
	}
}

func checkFalse(t *testing.T, cond bool) {
	t.Helper()
	if cond {
		t.Fatal("expected false")
	}
}

func checkNil(t *testing.T, v any) {
	t.Helper()
	if !isNil(v) {
		t.Fatalf("expected nil, got %v", v)
	}
}

func checkNotNil(t *testing.T, v any) {
	t.Helper()
	if isNil(v) {
		t.Fatal("expected non-nil")
	}
}

func checkLen(t *testing.T, v any, want int) {
	t.Helper()
	got, ok := valueLen(v)
	if !ok {
		t.Fatalf("expected value with length, got %T", v)
	}
	if got != want {
		t.Fatalf("want len %d, got %d", want, got)
	}
}

func checkEmpty(t *testing.T, v any) {
	t.Helper()
	if isNil(v) {
		return
	}
	if got, ok := valueLen(v); ok {
		if got != 0 {
			t.Fatalf("expected empty, got %v", v)
		}
		return
	}
	rv := reflect.ValueOf(v)
	if !reflect.DeepEqual(v, reflect.Zero(rv.Type()).Interface()) {
		t.Fatalf("expected empty, got %v", v)
	}
}

func checkContains(t *testing.T, container, item any) {
	t.Helper()
	if contains(container, item) {
		return
	}
	t.Fatalf("expected %v to contain %v", container, item)
}

func checkElementsMatch(t *testing.T, want, got any) {
	t.Helper()

	wantValue := reflect.ValueOf(want)
	gotValue := reflect.ValueOf(got)
	if !isList(wantValue) || !isList(gotValue) {
		t.Fatalf("expected list values, got %T and %T", want, got)
	}
	if wantValue.Len() != gotValue.Len() {
		t.Fatalf("want elements %v, got %v", want, got)
	}

	used := make([]bool, gotValue.Len())
	for i := range wantValue.Len() {
		found := false
		wantElem := wantValue.Index(i).Interface()
		for j := range gotValue.Len() {
			if used[j] {
				continue
			}
			if reflect.DeepEqual(wantElem, gotValue.Index(j).Interface()) {
				used[j] = true
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("want elements %v, got %v", want, got)
		}
	}
}

func checkInDelta(t *testing.T, want, got, delta any) {
	t.Helper()

	wantFloat, ok := toFloat64(want)
	if !ok {
		t.Fatalf("expected numeric want value, got %T", want)
	}
	gotFloat, ok := toFloat64(got)
	if !ok {
		t.Fatalf("expected numeric got value, got %T", got)
	}
	deltaFloat, ok := toFloat64(delta)
	if !ok {
		t.Fatalf("expected numeric delta value, got %T", delta)
	}
	if math.Abs(wantFloat-gotFloat) > deltaFloat {
		t.Fatalf("want %v, got %v", want, got)
	}
}

func isNil(v any) bool {
	if v == nil {
		return true
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return rv.IsNil()
	default:
		return false
	}
}

func valueLen(v any) (int, bool) {
	if v == nil {
		return 0, true
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Array, reflect.Chan, reflect.Map, reflect.Slice, reflect.String:
		return rv.Len(), true
	default:
		return 0, false
	}
}

func contains(container, item any) bool {
	containerValue := reflect.ValueOf(container)
	if !containerValue.IsValid() {
		return false
	}

	switch containerValue.Kind() {
	case reflect.String:
		itemString, ok := item.(string)
		return ok && strings.Contains(containerValue.String(), itemString)
	case reflect.Map:
		key := reflect.ValueOf(item)
		if !key.IsValid() {
			return false
		}
		keyType := containerValue.Type().Key()
		if key.Type().AssignableTo(keyType) {
			return containerValue.MapIndex(key).IsValid()
		}
		if key.Type().ConvertibleTo(keyType) {
			return containerValue.MapIndex(key.Convert(keyType)).IsValid()
		}
		return false
	case reflect.Array, reflect.Slice:
		for i := range containerValue.Len() {
			if reflect.DeepEqual(containerValue.Index(i).Interface(), item) {
				return true
			}
		}
	}
	return false
}

func isList(v reflect.Value) bool {
	if !v.IsValid() {
		return false
	}
	return v.Kind() == reflect.Array || v.Kind() == reflect.Slice
}

func toFloat64(v any) (float64, bool) {
	rv := reflect.ValueOf(v)
	if !rv.IsValid() {
		return 0, false
	}
	switch rv.Kind() {
	case reflect.Float32, reflect.Float64:
		return rv.Convert(reflect.TypeOf(float64(0))).Float(), true
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(rv.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return float64(rv.Uint()), true
	default:
		return 0, false
	}
}
