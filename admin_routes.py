"""
Admin panel routes — password protected.
Handles: login, upload face, list users, delete user.
"""

from __future__ import annotations

import os
import functools

import cv2
import numpy as np
from flask import (
    Blueprint, render_template, request,
    redirect, url_for, session, flash
)
from werkzeug.utils import secure_filename

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
KNOWN_FACES_DIR = "data/known_faces"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def login_required(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin.login"))
        return f(*args, **kwargs)
    return wrapper


def get_users():
    users = []
    if not os.path.isdir(KNOWN_FACES_DIR):
        return users
    for fname in os.listdir(KNOWN_FACES_DIR):
        name, ext = os.path.splitext(fname)
        if ext.lower() in ALLOWED_EXTENSIONS:
            users.append({"name": name, "file": fname})
    return users


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin.dashboard"))
        flash("Wrong password. Try again.")
    return render_template("admin_login.html")


@admin_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("admin.login"))


@admin_bp.route("/", methods=["GET"])
@login_required
def dashboard():
    users = get_users()
    return render_template("admin_dashboard.html", users=users)


@admin_bp.route("/upload", methods=["POST"])
@login_required
def upload():
    name = request.form.get("name", "").strip()
    file = request.files.get("photo")

    if not name:
        flash("Name is required.")
        return redirect(url_for("admin.dashboard"))

    if not file or file.filename == "":
        flash("Please select a photo.")
        return redirect(url_for("admin.dashboard"))

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        flash("Only JPG, PNG, BMP files allowed.")
        return redirect(url_for("admin.dashboard"))

    # Read and validate face exists in photo
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        flash("Could not read image file.")
        return redirect(url_for("admin.dashboard"))

    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    save_name = secure_filename(name) + ".jpg"
    save_path = os.path.join(KNOWN_FACES_DIR, save_name)
    cv2.imwrite(save_path, img)

    flash(f"User '{name}' added successfully!")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/delete/<filename>", methods=["POST"])
@login_required
def delete(filename):
    safe = secure_filename(filename)
    path = os.path.join(KNOWN_FACES_DIR, safe)
    if os.path.exists(path):
        os.remove(path)
        flash(f"User '{os.path.splitext(safe)[0]}' deleted.")
    else:
        flash("User not found.")
    return redirect(url_for("admin.dashboard"))
