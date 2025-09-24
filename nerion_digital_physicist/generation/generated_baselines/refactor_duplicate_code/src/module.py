def base_computation(x: int) -> int:
    return x * x + 1

def process_0(value: int):
    base = base_computation(value)
    return {"value": base}

def process_1(value: int):
    base = base_computation(value)
    return {"value": base}

def process_2(value: int):
    base = base_computation(value)
    return {"value": base}

def select_processor(index: int, value: int):
    options = [process_0, process_1, process_2]
    func = options[index % len(options)]
    return func(value)
