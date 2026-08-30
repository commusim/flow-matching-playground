const COLORS = [
  "#3b82f6",
  "#f97316",
  "#22c55e",
  "#ef4444",
  "#8b5cf6",
  "#a16207",
  "#ec4899",
  "#64748b",
  "#ca8a04",
  "#06b6d4",
];

const MODEL_LABELS = {
  additive: "入口加法条件CNN",
  adagn: "AdaGN条件CNN",
  latent: "VAE latent Flow",
  unet: "条件U-Net Flow",
  additive_condition: "入口加法条件CNN",
  adagn_condition: "AdaGN条件CNN",
  latent_flow: "VAE latent Flow",
  conditional_unet: "条件U-Net Flow",
};

const state = {
  flow: null,
  predictions: null,
  manifold: null,
  velocity: null,
  sprites: {},
};

function byId(id) {
  return document.getElementById(id);
}

function loadJson(path) {
  return fetch(path).then((response) => {
    if (!response.ok) {
      throw new Error(`无法加载 ${path}`);
    }
    return response.json();
  });
}

function loadImage(path) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = path;
  });
}

function clearCanvas(canvas, color = "#fbfcfe") {
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = color;
  context.fillRect(0, 0, canvas.width, canvas.height);
  return context;
}

function paddedExtent(pointCollections, padding = 0.08) {
  const points = pointCollections.flat();
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const dx = Math.max(maxX - minX, 1);
  const dy = Math.max(maxY - minY, 1);
  return [
    minX - dx * padding,
    maxX + dx * padding,
    minY - dy * padding,
    maxY + dy * padding,
  ];
}

function createMapper(canvas, extent, margin = 52) {
  const [minX, maxX, minY, maxY] = extent;
  return ([x, y]) => [
    margin + ((x - minX) / (maxX - minX)) * (canvas.width - margin * 2),
    canvas.height -
      margin -
      ((y - minY) / (maxY - minY)) * (canvas.height - margin * 2),
  ];
}

function drawAxes(context, canvas, extent, mapper, xLabel, yLabel) {
  context.strokeStyle = "#d5deea";
  context.lineWidth = 1;
  context.strokeRect(52, 28, canvas.width - 104, canvas.height - 80);
  context.fillStyle = "#5d6b80";
  context.font = "12px system-ui";
  context.textAlign = "center";
  context.fillText(xLabel, canvas.width / 2, canvas.height - 12);
  context.save();
  context.translate(14, canvas.height / 2);
  context.rotate(-Math.PI / 2);
  context.fillText(yLabel, 0, 0);
  context.restore();
  const [minX, maxX, minY, maxY] = extent;
  context.fillStyle = "#778397";
  context.font = "10px system-ui";
  context.fillText(
    minX.toFixed(1),
    mapper([minX, minY])[0],
    canvas.height - 34,
  );
  context.fillText(
    maxX.toFixed(1),
    mapper([maxX, minY])[0],
    canvas.height - 34,
  );
  context.textAlign = "right";
  context.fillText(minY.toFixed(1), 45, mapper([minX, minY])[1] + 4);
  context.fillText(maxY.toFixed(1), 45, mapper([minX, maxY])[1] + 4);
}

function drawArrow(context, start, change, color, scale = 1) {
  const end = [start[0] + change[0] * scale, start[1] + change[1] * scale];
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 1.2;
  context.beginPath();
  context.moveTo(start[0], start[1]);
  context.lineTo(end[0], end[1]);
  context.stroke();
  const angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
  const head = 4;
  context.beginPath();
  context.moveTo(end[0], end[1]);
  context.lineTo(
    end[0] - head * Math.cos(angle - Math.PI / 6),
    end[1] - head * Math.sin(angle - Math.PI / 6),
  );
  context.lineTo(
    end[0] - head * Math.cos(angle + Math.PI / 6),
    end[1] - head * Math.sin(angle + Math.PI / 6),
  );
  context.closePath();
  context.fill();
}

function drawFlow() {
  if (!state.flow) return;
  const mode = byId("flow-mode").value;
  const timeIndex = Number(byId("flow-time").value);
  const showArrows = byId("flow-arrows").checked;
  const canvas = byId("flow-canvas");
  const context = clearCanvas(canvas);
  const data = state.flow.modes[mode];
  const extent = [-3.1, 3.1, -2.8, 2.8];
  const mapper = createMapper(canvas, extent, 54);
  drawAxes(context, canvas, extent, mapper, "x₁", "x₂");

  context.fillStyle = "rgba(239,71,111,0.20)";
  data.target.forEach((point) => {
    const [x, y] = mapper(point);
    context.beginPath();
    context.arc(x, y, 2.4, 0, Math.PI * 2);
    context.fill();
  });

  if (showArrows) {
    const grid = state.flow.grid;
    const velocity = data.velocities[timeIndex];
    grid.forEach((point, index) => {
      const start = mapper(point);
      const vector = velocity[index];
      const magnitude = Math.hypot(vector[0], vector[1]);
      const normalizedScale = Math.min(0.18, 0.1 / Math.max(magnitude, 0.001));
      const mappedEnd = mapper([
        point[0] + vector[0] * normalizedScale,
        point[1] + vector[1] * normalizedScale,
      ]);
      drawArrow(
        context,
        start,
        [mappedEnd[0] - start[0], mappedEnd[1] - start[1]],
        "rgba(112,72,181,0.58)",
      );
    });
  }

  context.fillStyle = "rgba(49,87,213,0.78)";
  data.positions[timeIndex].forEach((point) => {
    const [x, y] = mapper(point);
    context.beginPath();
    context.arc(x, y, 3.1, 0, Math.PI * 2);
    context.fill();
  });

  const time = state.flow.times[timeIndex];
  byId("flow-time-value").textContent = time.toFixed(2);
  const descriptions = {
    unconditional_moons:
      "无条件模型必须从噪声自身决定最终位置。观察早期速度主要整理整体分布，中后期才形成双月牙。",
    conditional_moons:
      "条件0选择双月牙速度场。与条件1相比，网络参数相同，但条件Embedding改变了每个时刻的方向。",
    conditional_ring:
      "条件1选择圆环速度场。同一噪声在不同条件下会沿不同ODE轨迹进入不同目标分布。",
  };
  byId("flow-explanation").textContent = descriptions[mode];
}

function populateNumericSelect(select, count) {
  select.innerHTML = "";
  for (let index = 0; index < count; index += 1) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = String(index);
    select.appendChild(option);
  }
}

function drawMnist() {
  if (!state.predictions) return;
  const model = byId("mnist-model").value;
  const label = Number(byId("mnist-label").value);
  const timeIndex = Number(byId("mnist-time").value);
  const image = state.sprites[model];
  const canvas = byId("mnist-canvas");
  const context = clearCanvas(canvas, "#000000");
  context.imageSmoothingEnabled = false;
  context.drawImage(
    image,
    timeIndex * 28,
    label * 28,
    28,
    28,
    0,
    0,
    canvas.width,
    canvas.height,
  );

  const timeline = byId("mnist-timeline");
  const timelineContext = clearCanvas(timeline, "#000000");
  timelineContext.imageSmoothingEnabled = false;
  const cellWidth = timeline.width / 11;
  for (let index = 0; index < 11; index += 1) {
    const size = Math.min(cellWidth - 8, 72);
    const x = index * cellWidth + (cellWidth - size) / 2;
    const y = (timeline.height - size) / 2;
    timelineContext.drawImage(
      image,
      index * 28,
      label * 28,
      28,
      28,
      x,
      y,
      size,
      size,
    );
    if (index === timeIndex) {
      timelineContext.strokeStyle = "#79a0ff";
      timelineContext.lineWidth = 3;
      timelineContext.strokeRect(x - 2, y - 2, size + 4, size + 4);
    }
  }

  const record = state.predictions.models[model][timeIndex];
  const confidence = record.confidence[label];
  const predicted = confidence >= 0.8 ? record.predicted[label] : "unknown";
  byId("mnist-time-value").textContent =
    state.predictions.times[timeIndex].toFixed(2);
  byId("mnist-target").textContent = String(label);
  byId("mnist-prediction").textContent = String(predicted);
  byId("mnist-confidence").textContent = `${(confidence * 100).toFixed(1)}%`;

  const descriptions = {
    additive:
      "入口加法条件常被后续归一化削弱，图像虽然逐渐像数字，但目标Label未必控制全局结构。",
    adagn:
      "AdaGN在每个残差块注入条件，条件影响增强；但平坦CNN仍缺乏全局感受野。",
    latent:
      "VAE将像素压缩到8×7×7 latent。Decoder提供数字形状先验，因此比平坦像素CNN更容易形成完整笔画。",
    unet: "U-Net的7×7 Bottleneck规划全局数字拓扑，Skip Connection恢复细节，CFG进一步放大Label方向。",
  };
  byId("mnist-explanation").textContent = descriptions[model];
}

function getManifoldExtent() {
  const collections = [state.manifold.real.points];
  Object.values(state.manifold.models).forEach((model) => {
    model.forEach((time) => collections.push(time));
  });
  return paddedExtent(collections, 0.06);
}

function drawRealClusters(context, mapper, points, labels, alpha = 0.2) {
  const centroids = Array.from({ length: 10 }, () => [0, 0, 0]);
  points.forEach((point, index) => {
    const label = labels[index];
    const [x, y] = mapper(point);
    context.fillStyle = `${COLORS[label]}${Math.round(alpha * 255)
      .toString(16)
      .padStart(2, "0")}`;
    context.beginPath();
    context.arc(x, y, 2.2, 0, Math.PI * 2);
    context.fill();
    centroids[label][0] += point[0];
    centroids[label][1] += point[1];
    centroids[label][2] += 1;
  });
  context.font = "bold 13px system-ui";
  context.textAlign = "center";
  centroids.forEach((centroid, label) => {
    if (!centroid[2]) return;
    const point = [centroid[0] / centroid[2], centroid[1] / centroid[2]];
    const [x, y] = mapper(point);
    context.fillStyle = COLORS[label];
    context.fillText(String(label), x, y);
  });
}

function drawManifold() {
  if (!state.manifold) return;
  const model = byId("manifold-model").value;
  const label = Number(byId("manifold-label").value);
  const timeIndex = Number(byId("manifold-time").value);
  const canvas = byId("manifold-canvas");
  const context = clearCanvas(canvas);
  const extent = getManifoldExtent();
  const mapper = createMapper(canvas, extent, 60);
  drawAxes(context, canvas, extent, mapper, "shared t-SNE 1", "shared t-SNE 2");
  drawRealClusters(
    context,
    mapper,
    state.manifold.real.points,
    state.manifold.real.labels,
    0.2,
  );

  const expected = state.manifold.expected_labels;
  const modelPoints = state.manifold.models[model];
  const selectedIndices = expected
    .map((value, index) => (value === label ? index : -1))
    .filter((index) => index >= 0);

  context.strokeStyle = COLORS[label];
  context.lineWidth = 2;
  context.globalAlpha = 0.65;
  context.beginPath();
  for (let index = 0; index <= timeIndex; index += 1) {
    const centroid = [0, 0];
    selectedIndices.forEach((sample) => {
      centroid[0] += modelPoints[index][sample][0];
      centroid[1] += modelPoints[index][sample][1];
    });
    centroid[0] /= selectedIndices.length;
    centroid[1] /= selectedIndices.length;
    const [x, y] = mapper(centroid);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.stroke();
  context.globalAlpha = 1;

  modelPoints[timeIndex].forEach((point, index) => {
    if (expected[index] !== label) return;
    const [x, y] = mapper(point);
    context.fillStyle = COLORS[label];
    context.beginPath();
    context.arc(x, y, 5, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "white";
    context.lineWidth = 1;
    context.stroke();
  });
  byId("manifold-time-value").textContent =
    state.manifold.times[timeIndex].toFixed(2);
}

function getVelocityExtent() {
  const collections = [state.velocity.real.points];
  Object.values(state.velocity.models).forEach((model) => {
    model.centroids.forEach((time) => collections.push(time));
  });
  return paddedExtent(collections, 0.08);
}

function drawVelocity() {
  if (!state.velocity) return;
  const model = byId("velocity-model").value;
  const timeIndex = Number(byId("velocity-time").value);
  const canvas = byId("velocity-canvas");
  const context = clearCanvas(canvas);
  const extent = getVelocityExtent();
  const mapper = createMapper(canvas, extent, 60);
  drawAxes(
    context,
    canvas,
    extent,
    mapper,
    "classifier feature PCA 1",
    "classifier feature PCA 2",
  );
  drawRealClusters(
    context,
    mapper,
    state.velocity.real.points,
    state.velocity.real.labels,
    0.17,
  );

  const data = state.velocity.models[model];
  for (let label = 0; label < 10; label += 1) {
    const point = data.centroids[timeIndex][label];
    const velocity = data.velocity[timeIndex][label];
    const start = mapper(point);
    const next = mapper([
      point[0] + velocity[0] * 0.1,
      point[1] + velocity[1] * 0.1,
    ]);
    context.fillStyle = COLORS[label];
    context.beginPath();
    context.moveTo(start[0], start[1] - 7);
    context.lineTo(start[0] - 6, start[1] + 5);
    context.lineTo(start[0] + 6, start[1] + 5);
    context.closePath();
    context.fill();
    drawArrow(
      context,
      start,
      [next[0] - start[0], next[1] - start[1]],
      COLORS[label],
    );
    context.fillStyle = COLORS[label];
    context.font = "bold 11px system-ui";
    context.fillText(String(label), next[0] + 7, next[1] - 5);
  }
  const start = state.velocity.times[timeIndex];
  const end = state.velocity.times[timeIndex + 1];
  byId("velocity-time-value").textContent =
    `${start.toFixed(2)} → ${end.toFixed(2)}`;
}

function drawMetric() {
  if (!state.velocity) return;
  const metric = byId("metric-select").value;
  const canvas = byId("metric-canvas");
  const context = clearCanvas(canvas);
  const names = Object.keys(state.velocity.models);
  const valuesByModel = names.map(
    (name) => state.velocity.models[name][metric],
  );
  const times =
    metric === "distance"
      ? state.velocity.times
      : state.velocity.times.slice(0, -1).map((value, index) => {
          return (value + state.velocity.times[index + 1]) / 2;
        });
  const allValues = valuesByModel.flat();
  let minY = Math.min(...allValues);
  let maxY = Math.max(...allValues);
  if (metric === "alignment") {
    minY = Math.min(minY, 0);
    maxY = Math.max(maxY, 0);
  }
  const dy = Math.max(maxY - minY, 0.1);
  const extent = [0, 1, minY - dy * 0.08, maxY + dy * 0.08];
  const mapper = createMapper(canvas, extent, 58);
  const labels = {
    speed: "mean semantic speed ||df/dt||",
    alignment: "cosine alignment",
    distance: "distance to target centroid",
  };
  drawAxes(context, canvas, extent, mapper, "Flow time t", labels[metric]);
  if (metric === "alignment" && minY < 0 && maxY > 0) {
    const zeroStart = mapper([0, 0]);
    const zeroEnd = mapper([1, 0]);
    context.strokeStyle = "#98a4b5";
    context.beginPath();
    context.moveTo(...zeroStart);
    context.lineTo(...zeroEnd);
    context.stroke();
  }
  names.forEach((name, modelIndex) => {
    const color = ["#3157d5", "#f28e2b", "#2ca02c", "#d62728"][modelIndex];
    const values = valuesByModel[modelIndex];
    context.strokeStyle = color;
    context.lineWidth = 2.5;
    context.beginPath();
    values.forEach((value, index) => {
      const [x, y] = mapper([times[index], value]);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
    values.forEach((value, index) => {
      const [x, y] = mapper([times[index], value]);
      context.fillStyle = color;
      context.beginPath();
      context.arc(x, y, 3.5, 0, Math.PI * 2);
      context.fill();
    });
    context.fillStyle = color;
    context.font = "12px system-ui";
    context.textAlign = "left";
    context.fillText(MODEL_LABELS[name], 72 + modelIndex * 195, 24);
  });

  const explanations = {
    speed:
      "速度大不代表方向正确。应与目标方向对齐度共同观察：U-Net在生成中期既移动得快，也朝目标类别流形前进。",
    alignment:
      "大于0表示语义速度朝真实目标类别中心移动。后期变负不一定是失败，可能表示样本已进入正确类别流形并转向具体书写风格。",
    distance:
      "距离快速下降表示生成特征进入目标类别区域。若后期分类准确率保持很高但距离略回升，通常是在类别流形内部形成多样化样本。",
  };
  byId("metric-explanation").textContent = explanations[metric];
}

function populateModelSelect(select, names) {
  select.innerHTML = "";
  names.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = MODEL_LABELS[name] || name;
    select.appendChild(option);
  });
}

function attachInteractions() {
  ["flow-mode", "flow-time", "flow-arrows"].forEach((id) => {
    byId(id).addEventListener("input", drawFlow);
  });
  ["mnist-model", "mnist-label", "mnist-time"].forEach((id) => {
    byId(id).addEventListener("input", drawMnist);
  });
  ["manifold-model", "manifold-label", "manifold-time"].forEach((id) => {
    byId(id).addEventListener("input", drawManifold);
  });
  ["velocity-model", "velocity-time"].forEach((id) => {
    byId(id).addEventListener("input", drawVelocity);
  });
  byId("metric-select").addEventListener("input", drawMetric);
}

async function initialize() {
  try {
    const [
      flow,
      predictions,
      manifold,
      velocity,
      additive,
      adagn,
      latent,
      unet,
    ] = await Promise.all([
      loadJson("assets/flow_2d.json"),
      loadJson("assets/mnist_predictions.json"),
      loadJson("assets/semantic_tsne.json"),
      loadJson("assets/semantic_velocity.json"),
      loadImage("assets/mnist_additive_sprites.png"),
      loadImage("assets/mnist_adagn_sprites.png"),
      loadImage("assets/mnist_latent_sprites.png"),
      loadImage("assets/mnist_unet_sprites.png"),
    ]);
    state.flow = flow;
    state.predictions = predictions;
    state.manifold = manifold;
    state.velocity = velocity;
    state.sprites = { additive, adagn, latent, unet };

    populateNumericSelect(byId("mnist-label"), 10);
    populateNumericSelect(byId("manifold-label"), 10);
    populateModelSelect(byId("manifold-model"), Object.keys(manifold.models));
    populateModelSelect(byId("velocity-model"), Object.keys(velocity.models));
    byId("manifold-model").value = "conditional_unet";
    byId("velocity-model").value = "conditional_unet";
    attachInteractions();
    drawFlow();
    drawMnist();
    drawManifold();
    drawVelocity();
    drawMetric();
  } catch (error) {
    console.error(error);
    document.querySelectorAll(".dynamic-note").forEach((node) => {
      node.textContent =
        "交互数据加载失败。请通过GitHub Pages访问，或在docs目录启动本地HTTP服务器。";
    });
  }
}

window.addEventListener("DOMContentLoaded", initialize);
