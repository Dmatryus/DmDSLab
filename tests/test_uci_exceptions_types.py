"""
Тесты для исключений и типов UCI Dataset Loader.
"""

import contextlib

import pytest

from dmdslab.datasets.uci.uci_exceptions import (
    CacheError,
    ConfigurationError,
    DataFormatError,
    DatasetNotFoundError,
    NetworkError,
    UCIDatasetError,
    ValidationError,
)
from dmdslab.datasets.uci.uci_types import (
    DEFAULT_PICKLE_PROTOCOL,
    SUPPORTED_PICKLE_PROTOCOLS,
    CacheStatus,
    DatasetID,
    Domain,
    LogLevel,
    TaskType,
)


class TestTaskType:
    """Тесты для перечисления TaskType."""

    def test_task_type_values(self):
        """Тест значений TaskType."""
        assert TaskType.CLASSIFICATION.value == "classification"
        assert TaskType.REGRESSION.value == "regression"
        assert TaskType.CLUSTERING.value == "clustering"
        assert TaskType.UNKNOWN.value == "unknown"

    def test_from_string(self):
        """Тест создания TaskType из строки."""
        assert TaskType.from_string("classification") == TaskType.CLASSIFICATION
        assert TaskType.from_string("CLASSIFICATION") == TaskType.CLASSIFICATION
        assert TaskType.from_string(" Classification ") == TaskType.CLASSIFICATION

        assert TaskType.from_string("regression") == TaskType.REGRESSION
        assert TaskType.from_string("clustering") == TaskType.CLUSTERING
        assert TaskType.from_string("unknown") == TaskType.UNKNOWN

    def test_from_string_invalid(self):
        """Тест ошибки при неверном значении."""
        with pytest.raises(ValueError, match="Неизвестный тип задачи"):
            TaskType.from_string("invalid_type")

    def test_str_representation(self):
        """Тест строкового представления."""
        assert str(TaskType.CLASSIFICATION) == "classification"
        assert str(TaskType.REGRESSION) == "regression"


class TestDomain:
    """Тесты для перечисления Domain."""

    def test_domain_values(self):
        """Тест значений Domain."""
        assert Domain.BUSINESS.value == "business"
        assert Domain.COMPUTER.value == "computer"
        assert Domain.ENGINEERING.value == "engineering"
        assert Domain.GAMES.value == "games"
        assert Domain.LIFE.value == "life"
        assert Domain.PHYSICAL.value == "physical"
        assert Domain.SOCIAL.value == "social"
        assert Domain.OTHER.value == "other"

    def test_from_string(self):
        """Тест создания Domain из строки."""
        assert Domain.from_string("business") == Domain.BUSINESS
        assert Domain.from_string("LIFE") == Domain.LIFE
        assert Domain.from_string(" Computer ") == Domain.COMPUTER

    def test_from_string_default(self):
        """Тест значения по умолчанию для неизвестного домена."""
        assert Domain.from_string("unknown_domain") == Domain.OTHER
        assert Domain.from_string("") == Domain.OTHER

    def test_str_representation(self):
        """Тест строкового представления."""
        assert str(Domain.BUSINESS) == "business"
        assert str(Domain.LIFE) == "life"


class TestCacheStatus:
    """Тесты для перечисления CacheStatus."""

    def test_cache_status_values(self):
        """Тест значений CacheStatus."""
        assert CacheStatus.HIT.value == "hit"
        assert CacheStatus.MISS.value == "miss"
        assert CacheStatus.STALE.value == "stale"
        assert CacheStatus.CORRUPTED.value == "corrupted"

    def test_str_representation(self):
        """Тест строкового представления."""
        assert str(CacheStatus.HIT) == "hit"
        assert str(CacheStatus.MISS) == "miss"


class TestLogLevel:
    """Тесты для перечисления LogLevel."""

    def test_log_level_values(self):
        """Тест значений LogLevel."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"

    def test_numeric_level(self):
        """Тест получения числового уровня."""
        import logging

        assert LogLevel.DEBUG.numeric_level == logging.DEBUG
        assert LogLevel.INFO.numeric_level == logging.INFO
        assert LogLevel.WARNING.numeric_level == logging.WARNING
        assert LogLevel.ERROR.numeric_level == logging.ERROR

    def test_str_representation(self):
        """Тест строкового представления."""
        assert str(LogLevel.DEBUG) == "DEBUG"
        assert str(LogLevel.ERROR) == "ERROR"


class TestTypeAliases:
    """Тесты для псевдонимов типов."""

    def test_constants(self):
        """Тест констант."""
        assert DEFAULT_PICKLE_PROTOCOL == 4
        assert 4 in SUPPORTED_PICKLE_PROTOCOLS
        assert 5 in SUPPORTED_PICKLE_PROTOCOLS


class TestUCIDatasetError:
    """Тесты для базового класса исключений."""

    def test_basic_exception(self):
        """Тест базового исключения."""
        error = UCIDatasetError("Test error")

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.dataset_id is None
        assert error.error_type == "UCIDatasetError"
        assert error.details == {}

    def test_exception_with_dataset_id(self):
        """Тест исключения с ID датасета."""
        error = UCIDatasetError("Error loading", dataset_id=53)

        assert "Dataset ID: 53" in str(error)
        assert error.dataset_id == 53

    def test_exception_with_details(self):
        """Тест исключения с деталями."""
        details = {"url": "http://example.com", "status": 404}
        error = UCIDatasetError("Network error", details=details)

        assert "Details:" in str(error)
        assert "url=http://example.com" in str(error)
        assert "status=404" in str(error)

    def test_exception_repr(self):
        """Тест repr представления."""
        error = UCIDatasetError("Test", dataset_id=1, error_type="TestError")
        repr_str = repr(error)

        assert "UCIDatasetError" in repr_str
        assert "message='Test'" in repr_str
        assert "dataset_id=1" in repr_str
        assert "error_type='TestError'" in repr_str


class TestCacheError:
    """Тесты для CacheError."""

    def test_cache_error_basic(self):
        """Тест базового CacheError."""
        error = CacheError("Cache write failed")

        assert error.message == "Cache write failed"
        assert error.error_type == "CacheError"

    def test_cache_error_with_status(self):
        """Тест CacheError со статусом кеша."""
        error = CacheError(
            "Cache miss",
            dataset_id=53,
            cache_status=CacheStatus.MISS,
            cache_path="/path/to/cache",
        )

        assert error.cache_status == CacheStatus.MISS
        assert error.cache_path == "/path/to/cache"
        assert error.details["cache_status"] == "miss"
        assert error.details["cache_path"] == "/path/to/cache"


class TestValidationError:
    """Тесты для ValidationError."""

    def test_validation_error_basic(self):
        """Тест базового ValidationError."""
        error = ValidationError("Invalid data format")

        assert error.message == "Invalid data format"
        assert error.error_type == "ValidationError"

    def test_validation_error_with_field(self):
        """Тест ValidationError с информацией о поле."""
        error = ValidationError(
            "Invalid value",
            field_name="age",
            invalid_value=-5,
            expected_type="positive integer",
        )

        assert error.field_name == "age"
        assert error.invalid_value == -5
        assert error.expected_type == "positive integer"
        assert error.details["field_name"] == "age"
        assert error.details["invalid_value"] == "-5"
        assert error.details["expected_type"] == "positive integer"


class TestDatasetNotFoundError:
    """Тесты для DatasetNotFoundError."""

    def test_dataset_not_found_basic(self):
        """Тест базового DatasetNotFoundError."""
        error = DatasetNotFoundError(dataset_id=999)

        assert "Датасет с ID '999' не найден" in error.message
        assert error.dataset_id == 999

    def test_dataset_not_found_custom_message(self):
        """Тест с кастомным сообщением."""
        error = DatasetNotFoundError(dataset_id=999, message="Custom error message")

        assert error.message == "Custom error message"

    def test_dataset_not_found_with_locations(self):
        """Тест с местами поиска."""
        locations = ["cache", "UCI repository", "backup server"]
        error = DatasetNotFoundError(dataset_id=999, searched_locations=locations)

        assert error.searched_locations == locations
        assert error.details["searched_locations"] == locations


class TestNetworkError:
    """Тесты для NetworkError."""

    def test_network_error_basic(self):
        """Тест базового NetworkError."""
        error = NetworkError("Connection timeout")

        assert error.message == "Connection timeout"
        assert error.error_type == "NetworkError"

    def test_network_error_with_details(self):
        """Тест NetworkError с деталями."""
        error = NetworkError(
            "Failed to download",
            dataset_id=53,
            url="http://archive.ics.uci.edu/ml/...",
            status_code=503,
            retry_count=3,
        )

        assert error.url == "http://archive.ics.uci.edu/ml/..."
        assert error.status_code == 503
        assert error.retry_count == 3
        assert error.details["status_code"] == 503


class TestDataFormatError:
    """Тесты для DataFormatError."""

    def test_data_format_error_basic(self):
        """Тест базового DataFormatError."""
        error = DataFormatError("Unexpected data structure")

        assert error.message == "Unexpected data structure"
        assert error.error_type == "DataFormatError"

    def test_data_format_error_with_formats(self):
        """Тест DataFormatError с форматами."""
        error = DataFormatError(
            "Format mismatch",
            dataset_id=53,
            expected_format="CSV",
            actual_format="ARFF",
        )

        assert error.expected_format == "CSV"
        assert error.actual_format == "ARFF"
        assert error.details["expected_format"] == "CSV"
        assert error.details["actual_format"] == "ARFF"


class TestConfigurationError:
    """Тесты для ConfigurationError."""

    def test_configuration_error_basic(self):
        """Тест базового ConfigurationError."""
        error = ConfigurationError("Invalid configuration")

        assert error.message == "Invalid configuration"
        assert error.error_type == "ConfigurationError"

    def test_configuration_error_with_parameter(self):
        """Тест ConfigurationError с параметром."""
        error = ConfigurationError(
            "Invalid parameter value",
            parameter_name="cache_size",
            parameter_value=-100,
            valid_values=["positive integer", "> 0"],
        )

        assert error.parameter_name == "cache_size"
        assert error.parameter_value == -100
        assert error.valid_values == ["positive integer", "> 0"]
        assert error.details["parameter_name"] == "cache_size"
        assert error.details["parameter_value"] == "-100"


class TestExceptionHierarchy:
    """Тесты иерархии исключений."""

    def test_inheritance(self):
        """Тест наследования исключений."""
        # Все наследуются от UCIDatasetError
        assert issubclass(CacheError, UCIDatasetError)
        assert issubclass(ValidationError, UCIDatasetError)
        assert issubclass(DatasetNotFoundError, UCIDatasetError)
        assert issubclass(NetworkError, UCIDatasetError)
        assert issubclass(DataFormatError, UCIDatasetError)
        assert issubclass(ConfigurationError, UCIDatasetError)

        # UCIDatasetError наследуется от Exception
        assert issubclass(UCIDatasetError, Exception)

    def test_catch_base_exception(self):
        """Тест перехвата базового исключения."""
        try:
            raise CacheError("Test cache error")
        except UCIDatasetError as e:
            assert isinstance(e, CacheError)
            assert e.message == "Test cache error"

    def test_catch_specific_exception(self):
        """Тест перехвата конкретного исключения."""
        with pytest.raises(NetworkError):
            raise NetworkError("Network issue")

        # Не должно перехватываться другим типом
        with pytest.raises(UCIDatasetError):
            with contextlib.suppress(NetworkError):
                raise ValidationError("Validation issue")


if __name__ == "__main__":
    pytest.main([__file__])
