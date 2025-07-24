#!/bin/bash
# DmDSLab PyPI Publishing Script (Bash version)
# Версия: 2.0
# Автор: DmDSLab Team

set -e  # Остановка при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Параметры
TEST_REPO=false
SKIP_VERSION_CHECK=false
DRY_RUN=false

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-repo)
            TEST_REPO=true
            shift
            ;;
        --skip-version-check)
            SKIP_VERSION_CHECK=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Использование: $0 [OPTIONS]"
            echo "OPTIONS:"
            echo "  --test-repo              Публикация в TestPyPI"
            echo "  --skip-version-check     Пропустить проверку версии"
            echo "  --dry-run                Проверить без публикации"
            echo "  -h, --help               Показать эту справку"
            exit 0
            ;;
        *)
            echo -e "${RED}Неизвестный параметр: $1${NC}"
            exit 1
            ;;
    esac
done

# Конфигурация
PACKAGE_NAME="DmDSLab"
if [ "$TEST_REPO" = true ]; then
    PYPI_REPO="testpypi"
    PYPI_TOKEN="$TEST_PYPI_TOKEN"
    API_BASE="https://test.pypi.org"
else
    PYPI_REPO="pypi"
    PYPI_TOKEN="$PYPI_TOKEN"
    API_BASE="https://pypi.org"
fi

echo -e "${GREEN}DmDSLab PyPI Publishing Script${NC}"
echo -e "${GREEN}===============================${NC}"

# Проверка токена
if [ -z "$PYPI_TOKEN" ]; then
    TOKEN_VAR=$([ "$TEST_REPO" = true ] && echo "TEST_PYPI_TOKEN" || echo "PYPI_TOKEN")
    echo -e "${RED}ОШИБКА: Переменная окружения $TOKEN_VAR не установлена!${NC}"
    echo -e "${YELLOW}Установите токен: export $TOKEN_VAR='your-token-here'${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Токен найден${NC}"
echo -e "${GREEN}✓ Репозиторий: $PYPI_REPO${NC}"

# Очистка предыдущих сборок
echo -e "\n${YELLOW}Очистка предыдущих сборок...${NC}"
rm -rf build/ dist/ *.egg-info/

# Получение версии
echo -e "${YELLOW}Получение версии пакета...${NC}"
if ! VERSION=$(python -c "import $PACKAGE_NAME; print($PACKAGE_NAME.__version__)" 2>/dev/null); then
    echo -e "${RED}ОШИБКА: Не удалось получить версию пакета!${NC}"
    echo -e "${YELLOW}Убедитесь, что пакет установлен: pip install -e .${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Версия для публикации: $VERSION${NC}"

# Проверка существующей версии
if [ "$SKIP_VERSION_CHECK" = false ]; then
    echo -e "\n${YELLOW}Проверка существования версии на PyPI...${NC}"
    API_URL="$API_BASE/pypi/$PACKAGE_NAME/$VERSION/json"
    
    if curl -s -f "$API_URL" > /dev/null 2>&1; then
        echo -e "${RED}ОШИБКА: Версия $VERSION уже существует на $PYPI_REPO!${NC}"
        echo -e "${YELLOW}Обновите версию в setup.py и __init__.py${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ Версия $VERSION не найдена на $PYPI_REPO - можно публиковать${NC}"
    fi
else
    echo -e "\n${YELLOW}⚠ Пропускаем проверку версии по запросу${NC}"
fi

if [ "$DRY_RUN" = true ]; then
    echo -e "\n${CYAN}🔍 РЕЖИМ DRY RUN - публикация не будет выполнена${NC}"
fi

# Обновление инструментов сборки
echo -e "\n${YELLOW}Обновление инструментов сборки...${NC}"
python -m pip install --upgrade pip build twine

# Сборка пакета
echo -e "\n${YELLOW}Сборка пакета...${NC}"
python -m build
echo -e "${GREEN}✓ Пакет собран${NC}"

# Проверка пакета
echo -e "\n${YELLOW}Проверка пакета...${NC}"
twine check dist/*
echo -e "${GREEN}✓ Пакет прошел проверку${NC}"

# Публикация
if [ "$DRY_RUN" = false ]; then
    echo -e "\n${YELLOW}Публикация на $PYPI_REPO...${NC}"
    
    twine upload \
        --repository "$PYPI_REPO" \
        --username "__token__" \
        --password "$PYPI_TOKEN" \
        dist/*
    
    echo -e "\n${GREEN}🎉 УСПЕШНО ОПУБЛИКОВАНО!${NC}"
    PROJECT_URL="$API_BASE/project/$PACKAGE_NAME/$VERSION"
    echo -e "${GREEN}📦 Ссылка: $PROJECT_URL${NC}"
    
    if [ "$TEST_REPO" = false ]; then
        echo -e "\n${CYAN}📋 Для установки:${NC}"
        echo -e "${NC}pip install $PACKAGE_NAME==$VERSION${NC}"
    fi
else
    echo -e "\n${CYAN}✓ DRY RUN завершен - все проверки пройдены${NC}"
    echo -e "${YELLOW}Для реальной публикации запустите без флага --dry-run${NC}"
fi

# Очистка временных файлов
echo -e "\n${YELLOW}Очистка временных файлов...${NC}"
rm -rf build/ *.egg-info/
echo -e "${GREEN}✓ Очистка завершена${NC}"

echo -e "\n${GREEN}✨ Скрипт завершен успешно!${NC}"