from typing import List
from abc import ABC, abstractmethod


class DataProcessor(ABC):
    """Abstract base class for data processing operations."""
    
    @abstractmethod
    def process(self, data: List[int]) -> int:
        """Process the data and return the result."""
        pass


class FilteredSumProcessor(DataProcessor):
    """Processes data by filtering values and calculating their doubled sum."""
    
    MIN_VALUE = 10
    MAX_VALUE = 100
    MULTIPLIER = 2
    
    def process(self, data: List[int]) -> int:
        """
        Filter data for values between MIN_VALUE and MAX_VALUE (exclusive),
        double them, and return their sum.
        """
        filtered_values = self._filter_values(data)
        doubled_values = self._double_values(filtered_values)
        total_sum = self._calculate_sum(doubled_values)
        self._display_result(total_sum)
        return total_sum
    
    def _filter_values(self, data: List[int]) -> List[int]:
        """Filter values that are greater than MIN_VALUE and less than MAX_VALUE."""
        return [value for value in data if self.MIN_VALUE < value < self.MAX_VALUE]
    
    def _double_values(self, values: List[int]) -> List[int]:
        """Double each value in the list."""
        return [value * self.MULTIPLIER for value in values]
    
    def _calculate_sum(self, values: List[int]) -> int:
        """Calculate the sum of all values."""
        return sum(values)
    
    def _display_result(self, result: int) -> None:
        """Display the result."""
        print(f"Result is: {result}")


class DataManager:
    """Manages data collection and processing."""
    
    def __init__(self, processor: DataProcessor = None):
        self._data: List[int] = []
        self._processor = processor or FilteredSumProcessor()
    
    def add_value(self, value: int) -> None:
        """Add a value to the data collection."""
        self._data.append(value)
    
    def process_data(self) -> int:
        """Process the collected data using the configured processor."""
        return self._processor.process(self._data)


def process_data(data: List[int]) -> int:
    """Legacy function for backward compatibility."""
    processor = FilteredSumProcessor()
    return processor.process(data)