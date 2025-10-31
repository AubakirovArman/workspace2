"""Blueprint with HTML rendering routes."""
from __future__ import annotations

from flask import Blueprint, redirect, render_template, url_for

pages_bp = Blueprint('pages', __name__)


@pages_bp.route('/')
def index():
    return redirect(url_for('pages.realtime'))


@pages_bp.route('/realtime')
def realtime():
    return render_template('realtime.html')
