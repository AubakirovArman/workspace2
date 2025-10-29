#!/bin/bash
# Avatar Lipsync Service Launcher
# Запускает приложение как фоновый процесс

APP_DIR="/workspace"
APP_FILE="app.py"
PID_FILE="$APP_DIR/app.pid"
LOG_FILE="$APP_DIR/app.log"

case "$1" in
    start)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "❌ Служба уже запущена (PID: $PID)"
                exit 1
            fi
        fi
        
        echo "🚀 Запуск Avatar Lipsync Service..."
        cd "$APP_DIR"
        nohup python3 "$APP_FILE" > "$LOG_FILE" 2>&1 &
        PID=$!
        echo $PID > "$PID_FILE"
        echo "✅ Служба запущена (PID: $PID)"
        echo "📝 Логи: $LOG_FILE"
        echo "🌐 URL: http://localhost:3000"
        ;;
        
    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "❌ PID файл не найден. Служба не запущена?"
            exit 1
        fi
        
        PID=$(cat "$PID_FILE")
        echo "🛑 Остановка службы (PID: $PID)..."
        
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID
            sleep 2
            
            # Проверяем, завершился ли процесс
            if ps -p $PID > /dev/null 2>&1; then
                echo "⚠️ Процесс не завершился, принудительная остановка..."
                kill -9 $PID
            fi
            
            rm -f "$PID_FILE"
            echo "✅ Служба остановлена"
        else
            echo "⚠️ Процесс не найден, очистка PID файла"
            rm -f "$PID_FILE"
        fi
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "✅ Служба работает (PID: $PID)"
                echo ""
                echo "📊 Информация о процессе:"
                ps -p $PID -o pid,vsz,rss,pcpu,etime,cmd
                echo ""
                echo "📝 Последние 10 строк лога:"
                tail -10 "$LOG_FILE"
            else
                echo "❌ PID файл существует, но процесс не запущен"
            fi
        else
            echo "❌ Служба не запущена"
        fi
        ;;
        
    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "❌ Лог файл не найден"
        fi
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Команды:"
        echo "  start    - Запустить службу"
        echo "  stop     - Остановить службу"
        echo "  restart  - Перезапустить службу"
        echo "  status   - Проверить статус службы"
        echo "  logs     - Показать логи в реальном времени"
        exit 1
        ;;
esac

exit 0
