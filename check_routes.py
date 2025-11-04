#!/usr/bin/env python3
"""Проверка зарегистрированных роутов"""
import sys
sys.path.insert(0, '/home/arman/workspace2/modern-lipsync')

from app_core import create_app

app = create_app()

print("\n" + "="*70)
print("Все зарегистрированные роуты:")
print("="*70)

routes = []
for rule in app.url_map.iter_rules():
    routes.append({
        'endpoint': rule.endpoint,
        'methods': ','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'})),
        'path': rule.rule
    })

# Сортируем по пути
routes.sort(key=lambda x: x['path'])

for route in routes:
    print(f"{route['path']:40} [{route['methods']:15}] -> {route['endpoint']}")

print("="*70)

# Проверяем конкретно mjpeg
mjpeg_routes = [r for r in routes if 'mjpeg' in r['path'].lower()]
print(f"\nMJPEG роуты найдено: {len(mjpeg_routes)}")
for route in mjpeg_routes:
    print(f"  ✓ {route['path']} [{route['methods']}]")

if not mjpeg_routes:
    print("  ❌ MJPEG роуты НЕ найдены!")
