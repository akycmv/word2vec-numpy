import struct

from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return """<!DOCTYPE html>
<html>
<head>
  <title>loss</title>
</head>
<body>
  <p>word2vec / loss</p>
  <p>
    pts: <b id="v-pts">-</b> &nbsp;
    latest: <b id="v-latest">-</b> &nbsp;
    min: <b id="v-min">-</b> &nbsp;
    avg(20): <b id="v-avg">-</b>
  </p>
  <p>
    smoothing window:
    <input type="range" id="k-w" min="2" max="100" value="20" style="vertical-align:middle">
    <input type="number" id="k-w-num" min="2" max="100" value="20" style="width:50px">
  </p>
  <canvas id="c" width="1200" height="400" style="border:1px solid #ccc"></canvas>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    let data = [], smoothW = 20;

    const chart = new Chart(document.getElementById('c'), {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'smooth', data: [], borderColor: '#000', borderWidth: 1, pointRadius: 0, tension: 0.4, fill: false }
      ]},
      options: {
        animation: false, responsive: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { maxTicksLimit: 10 } },
          y: { min: 0 }
        }
      }
    });

    const smooth = (arr, w) => arr.map((_, i) => {
      const s = arr.slice(Math.max(0, i - w), i + 1);
      return s.reduce((a, b) => a + b, 0) / s.length;
    });

    function render() {
      if (!data.length) return;
      const sm = smooth(data, smoothW);
      chart.data.labels = data.map((_, i) => i * 50 + 'k');
      chart.data.datasets[0].data = sm;
      chart.update('none');
      document.getElementById('v-pts').textContent    = data.length;
      document.getElementById('v-latest').textContent = data.at(-1).toFixed(3);
      document.getElementById('v-min').textContent    = Math.min(...data).toFixed(3);
      const tail = sm.slice(-20);
      document.getElementById('v-avg').textContent    = (tail.reduce((a,b)=>a+b,0)/tail.length).toFixed(3);
    }

    function setWindow(v) {
      smoothW = Math.min(100, Math.max(2, +v || 2));
      document.getElementById('k-w').value     = smoothW;
      document.getElementById('k-w-num').value = smoothW;
      render();
    }

    document.getElementById('k-w').oninput     = e => setWindow(e.target.value);
    document.getElementById('k-w-num').oninput = e => setWindow(e.target.value);

    setInterval(async () => { data = await fetch('/data').then(r => r.json()); render(); }, 2000);
  </script>
</body>
</html>"""


@app.route("/data")
def data():
    try:
        with open("loss.bin", "rb") as f:
            raw = f.read()
        return jsonify(list(struct.unpack(f"{len(raw) // 4}f", raw)))
    except:
        return jsonify([])


app.run(port=5000)
