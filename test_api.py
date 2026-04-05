"""
Test script for GLM-OCR API
Run the server first: uvicorn main:app --reload
Then run: python test_api.py
"""

import requests
import json
import os

BASE_URL = "http://localhost:8000"


def print_result(label: str, response: requests.Response):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Status Code : {response.status_code}")
    try:
        data = response.json()
        print(f"  Response    :\n{json.dumps(data, indent=4, ensure_ascii=False)}")
    except Exception:
        print(f"  Response    : {response.text}")


# ── Test 1: GET / ─────────────────────────────────────────────────────────────
def test_root():
    res = requests.get(f"{BASE_URL}/")
    print_result("TEST 1 — GET /  (API Introduction)", res)
    assert res.status_code == 200, "Root endpoint failed!"


# ── Test 2: GET /health ───────────────────────────────────────────────────────
def test_health():
    res = requests.get(f"{BASE_URL}/health")
    print_result("TEST 2 — GET /health  (Health Check)", res)
    assert res.status_code == 200, "Health endpoint failed!"
    assert res.json()["model_loaded"] is True, "Model not loaded!"


# ── Test 3: POST /predict/url — simple text ───────────────────────────────────
def test_predict_url_text():
    payload = {
        "url": "https://funteacherfiles.com/wp-content/uploads/2022/10/Slide2-3.jpg",
        "mode": "text",
    }
    res = requests.post(f"{BASE_URL}/predict/url", json=payload)
    print_result("TEST 3 — POST /predict/url  (mode=text)", res)
    assert res.status_code == 200, "URL predict (text mode) failed!"


# ── Test 4: POST /predict/url — document mode ────────────────────────────────
def test_predict_url_document():
    # Use an image that actually has text/document content
    payload = {
        "url": "https://www.w3.org/WAI/WCAG21/Techniques/pdf/img/table-word.jpg",
        "mode": "document",
    }
    res = requests.post(f"{BASE_URL}/predict/url", json=payload)
    print_result("TEST 4 — POST /predict/url  (mode=document)", res)
    assert res.status_code == 200, "URL predict (document mode) failed!"


# ── Test 5: POST /predict/upload — upload local file ─────────────────────────
def test_predict_upload():
    # Create a simple test image if it doesn't exist
    test_img_path = "test_image.png"

    if not os.path.exists(test_img_path):
        # Download a sample image for testing
        sample_url = "https://www.w3.org/WAI/WCAG21/Techniques/pdf/img/table-word.jpg"
        img_data = requests.get(sample_url).content
        with open(test_img_path, "wb") as f:
            f.write(img_data)
        print(f"\n[INFO] Downloaded test image to {test_img_path}")

    with open(test_img_path, "rb") as f:
        files = {"file": (test_img_path, f, "image/png")}
        data = {"mode": "text"}
        res = requests.post(f"{BASE_URL}/predict/upload", files=files, data=data)

    print_result("TEST 5 — POST /predict/upload  (local file)", res)
    assert res.status_code == 200, "Upload predict failed!"


# ── Test 6: Error cases ───────────────────────────────────────────────────────
def test_error_invalid_url():
    payload = {"url": "not-a-valid-url", "mode": "text"}
    res = requests.post(f"{BASE_URL}/predict/url", json=payload)
    print_result("TEST 6 — Error: Invalid URL format (expect 422)", res)
    assert res.status_code == 422, "Should return 422 for invalid URL!"


def test_error_missing_body():
    res = requests.post(f"{BASE_URL}/predict/url", json={})
    print_result("TEST 7 — Error: Missing request body (expect 422)", res)
    assert res.status_code == 422, "Should return 422 for missing body!"


# ── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   GLM-OCR API — Test Suite")
    print("   Server:", BASE_URL)
    print("="*60)

    tests = [
        ("Root endpoint",           test_root),
        ("Health check",            test_health),
        ("Predict URL (text)",      test_predict_url_text),
        ("Predict URL (document)",  test_predict_url_document),
        ("Predict upload (file)",   test_predict_upload),
        ("Error: invalid URL",      test_error_invalid_url),
        ("Error: missing body",     test_error_missing_body),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")