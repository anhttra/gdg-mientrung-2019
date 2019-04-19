# How to run demo

## Requirements:
* Python3
* Virtualenv
* Ubuntu


You better create virtualenv to run this demo
```bash
virtualenv -p python3 venv
source venv/bin/activate
```

Install requirement packages:
```bash
pip install -r requirements.txt
```

Start application using flask
```bash
flask run
```

If you want to run on debug-environment
```bash
export FLASK_ENV=development
flask run
```
