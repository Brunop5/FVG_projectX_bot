def ema(data, length):
    # returns a single number
    pass

def sma(bars, length):
    # returns a single number
    pass

def get_atr(data, length):
    pass

def crossover(current_value, previous_value, threshold):
    return current_value > threshold and previous_value <= threshold

def crossunder(current_value, previous_value, threshold):
    return current_value < threshold and previous_value >= threshold