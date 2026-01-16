// ================================================================
// MATH UTILITIES
// ================================================================

function safeLog(x, base) {
    const val = x <= 0 ? 1e-15 : x;
    return base ? Math.log(val) / Math.log(base) : Math.log(val);
}

function safeLog2(x) { return safeLog(x, 2); }
function safeLog10(x) { return safeLog(x, 10); }

function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const ez = Math.exp(z);
    return ez / (1 + ez);
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
}

function clip(x, min, max) { return Math.max(min, Math.min(max, x)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function mapRange(x, inMin, inMax, outMin, outMax) { return outMin + (x - inMin) * (outMax - outMin) / (inMax - inMin); }

function gaussianPDF(x, mu, sigma) {
    const z = (x - mu) / sigma;
    return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

function randomNormal(mean = 0, std = 1) {
    const u1 = Math.random(), u2 = Math.random();
    return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ================================================================
// STATISTICS
// ================================================================

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }

function std(arr) {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / arr.length);
}

function entropy(probs, base = Math.E) {
    let H = 0;
    for (const p of probs) if (p > 0) H -= p * safeLog(p, base);
    return H;
}

function crossEntropy(pTrue, qModel, base = Math.E) {
    let H = 0;
    for (let i = 0; i < pTrue.length; i++) {
        if (pTrue[i] > 0) H -= pTrue[i] * safeLog(qModel[i], base);
    }
    return H;
}

function klDivergence(pTrue, qModel, base = Math.E) {
    return crossEntropy(pTrue, qModel, base) - entropy(pTrue, base);
}

// ================================================================
// CANVAS UTILITIES
// ================================================================

function createCoordSystem(canvas, bounds, padding = { top: 40, right: 30, bottom: 50, left: 60 }) {
    const W = canvas.width, H = canvas.height;
    const plotW = W - padding.left - padding.right;
    const plotH = H - padding.top - padding.bottom;
    
    return {
        toCanvas(x, y) {
            const cx = padding.left + (x - bounds.xMin) / (bounds.xMax - bounds.xMin) * plotW;
            const cy = padding.top + (1 - (y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * plotH;
            return { cx, cy };
        },
        toData(cx, cy) {
            const x = bounds.xMin + (cx - padding.left) / plotW * (bounds.xMax - bounds.xMin);
            const y = bounds.yMax - (cy - padding.top) / plotH * (bounds.yMax - bounds.yMin);
            return { x, y };
        },
        drawAxes(ctx, options = {}) {
            const { xLabel = 'x', yLabel = 'y', grid = true } = options;
            ctx.strokeStyle = '#e5e4e2';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(padding.left, padding.top);
            ctx.lineTo(padding.left, H - padding.bottom);
            ctx.lineTo(W - padding.right, H - padding.bottom);
            ctx.stroke();
            
            ctx.fillStyle = '#6b7280';
            ctx.font = '12px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(xLabel, padding.left + plotW / 2, H - 10);
            ctx.save();
            ctx.translate(15, padding.top + plotH / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(yLabel, 0, 0);
            ctx.restore();
        },
        padding, plotW, plotH, W, H
    };
}

function drawLineChart(ctx, coords, points, options = {}) {
    const { color = '#c45d4a', lineWidth = 2.5 } = options;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    points.forEach((pt, i) => {
        const { cx, cy } = coords.toCanvas(pt.x, pt.y);
        i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    });
    ctx.stroke();
}

function drawScatter(ctx, coords, points, options = {}) {
    const { color = 'rgba(0,0,0,0.5)', radius = 3 } = options;
    ctx.fillStyle = color;
    points.forEach(pt => {
        const { cx, cy } = coords.toCanvas(pt.x, pt.y);
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fill();
    });
}

function drawPoint(ctx, coords, x, y, options = {}) {
    const { color = '#c45d4a', radius = 6 } = options;
    const { cx, cy } = coords.toCanvas(x, y);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
}

// ================================================================
// ANIMATION
// ================================================================

function createAnimationLoop(callback) {
    let running = false;
    let animationId = null;
    
    function loop(timestamp) {
        if (!running) return;
        callback(timestamp);
        animationId = requestAnimationFrame(loop);
    }
    
    return {
        start() { if (!running) { running = true; animationId = requestAnimationFrame(loop); } },
        stop() { running = false; if (animationId) cancelAnimationFrame(animationId); },
        get running() { return running; }
    };
}

function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

// ================================================================
// UI UTILITIES
// ================================================================

function formatNumber(n, decimals = 2) { return n.toFixed(decimals); }
function updateText(id, text) { document.getElementById(id).textContent = text; }
function getSliderValue(id, scale = 1) { return parseFloat(document.getElementById(id).value) * scale; }

function setupToggleGroup(buttonIds, callback) {
    buttonIds.forEach(id => {
        document.getElementById(id).addEventListener('click', () => {
            buttonIds.forEach(bid => document.getElementById(bid).classList.remove('active'));
            document.getElementById(id).classList.add('active');
            callback(id);
        });
    });
}

// ================================================================
// MATRIX CLASS
// ================================================================

class Matrix {
    constructor(rows, cols, data = null) {
        this.rows = rows;
        this.cols = cols;
        this.data = data || new Float64Array(rows * cols);
    }
    
    static zeros(rows, cols) { return new Matrix(rows, cols); }
    
    static random(rows, cols, scale = 1) {
        const m = new Matrix(rows, cols);
        for (let i = 0; i < m.data.length; i++) m.data[i] = (Math.random() * 2 - 1) * scale;
        return m;
    }
    
    get(i, j) { return this.data[i * this.cols + j]; }
    set(i, j, v) { this.data[i * this.cols + j] = v; }
    
    add(o) {
        const r = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) r.data[i] = this.data[i] + o.data[i];
        return r;
    }
    
    sub(o) {
        const r = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) r.data[i] = this.data[i] - o.data[i];
        return r;
    }
    
    scale(s) {
        const r = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) r.data[i] = this.data[i] * s;
        return r;
    }
    
    static matmul(a, b) {
        const r = new Matrix(a.rows, b.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < b.cols; j++) {
                let s = 0;
                for (let k = 0; k < a.cols; k++) s += a.get(i, k) * b.get(k, j);
                r.set(i, j, s);
            }
        }
        return r;
    }
    
    static transpose(m) {
        const r = new Matrix(m.cols, m.rows);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) r.set(j, i, m.get(i, j));
        }
        return r;
    }
    
    addBias(b) {
        const r = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) r.set(i, j, this.get(i, j) + b.get(0, j));
        }
        return r;
    }
    
    clone() {
        const r = new Matrix(this.rows, this.cols);
        r.data.set(this.data);
        return r;
    }
}
