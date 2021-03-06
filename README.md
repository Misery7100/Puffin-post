# Puffin-post

Скрипт для постпроцессинга результатов вычислений, проведенных при помощи кода [Puffin](https://github.com/UKFELs/Puffin)

## Использование

1. Проверить, что установлен `python>=3.7`
2. Установить зависимости 
   ```bash
   pip install -r requirements.txt
   ```
3. Скопировать путь папки с результатами вычисления Puffin (например, `/home/user/Puffin/inputs/simple/1D/CLARA`)
4. Проверить basename файлов (оно же - имя основного input файла для вычислений) (например, `clara`)
5. Из папки со скриптом `puffin_postprocess.py` произвести выполнение постпроцессинга:

    ```bash 
    python puffin_postprocess.py -bn clara -d /home/user/Puffin/inputs/simple/1D/CLARA
    ```
6. Проверить output файлы в формате .dat в директории post_output внутри папки с результатами вычислений (в данном случае
`/home/user/Puffin/inputs/simple/1D/CLARA/post_output`)

## Заметки

- Выполнение каких-либо других скриптов не требуется, powPrep.py из основного репозитория интегрирован в текущий скрипт постпроцессинга;
- output .dat файлы достаточно большие (например, табличка для электронной плотности у меня почти Гб весила), т.к. такой формат для хранения подобных данных неоптимален.
