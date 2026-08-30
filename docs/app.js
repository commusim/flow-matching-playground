const COLORS = [
  "#2878b5",
  "#e6862f",
  "#3c8d85",
  "#b3242c",
  "#7656a6",
  "#9b6b55",
  "#d66a9a",
  "#7a7f84",
  "#c49a36",
  "#42a7b3",
];

const MODEL_LABELS = {
  additive: "入口加法CNN",
  adagn: "AdaGN CNN",
  latent: "VAE latent Flow",
  unet: "条件U-Net",
  additive_condition: "入口加法CNN",
  adagn_condition: "AdaGN CNN",
  latent_flow: "VAE latent Flow",
  conditional_unet: "条件U-Net",
};

const state = {
  flow: null,
  predictions: null,
  manifold: null,
  velocity: null,
  sprites: {},
  conditionGrids: {},
};

function byId(id) {
  return document.getElementById(id);
}

function loadJson(path) {
  return fetch(path).then((response) => {
    if (!response.ok) throw new Error(`无法加载 ${path}`);
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

function clearCanvas(canvas, color = "#ffffff") {
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = color;
  context.fillRect(0, 0, canvas.width, canvas.height);
  return context;
}

function paddedExtent(collections, padding = 0.08) {
  const points = collections.flat();
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

function mapperForRect(rect, extent) {
  const [minX, maxX, minY, maxY] = extent;
  return ([x, y]) => [
    rect.x + ((x - minX) / (maxX - minX)) * rect.width,
    rect.y + rect.height - ((y - minY) / (maxY - minY)) * rect.height,
  ];
}

function createMapper(canvas, extent, margin = 58) {
  return mapperForRect(
    {
      x: margin,
      y: 28,
      width: canvas.width - margin * 2,
      height: canvas.height - 82,
    },
    extent,
  );
}

function drawAxes(context, canvas, extent, mapper, xLabel, yLabel) {
  context.strokeStyle = "#bfc3c6";
  context.lineWidth = 1;
  context.strokeRect(58, 28, canvas.width - 116, canvas.height - 82);
  context.fillStyle = "#4d5155";
  context.font = "12px Arial";
  context.textAlign = "center";
  context.fillText(xLabel, canvas.width / 2, canvas.height - 12);
  context.save();
  context.translate(15, canvas.height / 2);
  context.rotate(-Math.PI / 2);
  context.fillText(yLabel, 0, 0);
  context.restore();
  const [minX, maxX, minY, maxY] = extent;
  context.fillStyle = "#777";
  context.font = "10px Arial";
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
  context.fillText(minY.toFixed(1), 50, mapper([minX, minY])[1] + 4);
  context.fillText(maxY.toFixed(1), 50, mapper([minX, maxY])[1] + 4);
}

function drawArrow(context, start, change, color, head = 4, width = 1.15) {
  const end = [start[0] + change[0], start[1] + change[1]];
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = width;
  context.beginPath();
  context.moveTo(start[0], start[1]);
  context.lineTo(end[0], end[1]);
  context.stroke();
  const angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
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

function drawFlowContent(
  context,
  rect,
  data,
  grid,
  timeIndex,
  showArrows,
  title,
) {
  const extent = [-3.1, 3.1, -2.8, 2.8];
  const mapper = mapperForRect(rect, extent);
  context.strokeStyle = "#bfc3c6";
  context.strokeRect(rect.x, rect.y, rect.width, rect.height);
  context.fillStyle = "#222";
  context.font = "bold 13px Arial";
  context.textAlign = "left";
  context.fillText(title, rect.x + 8, rect.y + 18);

  context.fillStyle = "rgba(179,36,44,0.15)";
  data.target.forEach((point) => {
    const [x, y] = mapper(point);
    context.beginPath();
    context.arc(x, y, 2.7, 0, Math.PI * 2);
    context.fill();
  });

  if (showArrows) {
    const velocity = data.velocities[timeIndex];
    grid.forEach((point, index) => {
      const vector = velocity[index];
      const magnitude = Math.hypot(vector[0], vector[1]);
      const scale = Math.min(0.18, 0.1 / Math.max(magnitude, 0.001));
      const start = mapper(point);
      const end = mapper([
        point[0] + vector[0] * scale,
        point[1] + vector[1] * scale,
      ]);
      drawArrow(
        context,
        start,
        [end[0] - start[0], end[1] - start[1]],
        "rgba(60,90,130,0.46)",
        3,
      );
    });
  }

  context.fillStyle = "rgba(40,120,181,0.82)";
  data.positions[timeIndex].forEach((point) => {
    const [x, y] = mapper(point);
    context.beginPath();
    context.arc(x, y, 2.8, 0, Math.PI * 2);
    context.fill();
  });
}

function drawUnconditional() {
  if (!state.flow) return;
  const timeIndex = Number(byId("unconditional-time").value);
  const showArrows = byId("unconditional-arrows").checked;
  const canvas = byId("unconditional-canvas");
  const context = clearCanvas(canvas);
  const extent = [-3.1, 3.1, -2.8, 2.8];
  const mapper = createMapper(canvas, extent, 58);
  drawAxes(context, canvas, extent, mapper, "x₁", "x₂");
  drawFlowContent(
    context,
    { x: 58, y: 28, width: canvas.width - 116, height: canvas.height - 82 },
    state.flow.modes.unconditional_moons,
    state.flow.grid,
    timeIndex,
    showArrows,
    "unconditional field",
  );
  byId("unconditional-time-value").textContent =
    state.flow.times[timeIndex].toFixed(3);

  const stages = byId("unconditional-stages");
  const stagesContext = clearCanvas(stages);
  const indices = [
    0,
    Math.floor((state.flow.times.length - 1) / 2),
    state.flow.times.length - 1,
  ];
  const gap = 22;
  const width = (stages.width - gap * 4) / 3;
  indices.forEach((index, panel) => {
    drawFlowContent(
      stagesContext,
      {
        x: gap + panel * (width + gap),
        y: 32,
        width,
        height: stages.height - 58,
      },
      state.flow.modes.unconditional_moons,
      state.flow.grid,
      index,
      panel === 1,
      `t = ${state.flow.times[index].toFixed(2)}`,
    );
  });
}

function drawConditional() {
  if (!state.flow) return;
  const mode = byId("conditional-mode").value;
  const timeIndex = Number(byId("conditional-time").value);
  const showArrows = byId("conditional-arrows").checked;
  const canvas = byId("conditional-canvas");
  const context = clearCanvas(canvas);
  const extent = [-3.1, 3.1, -2.8, 2.8];
  const mapper = createMapper(canvas, extent, 58);
  drawAxes(context, canvas, extent, mapper, "x₁", "x₂");
  drawFlowContent(
    context,
    { x: 58, y: 28, width: canvas.width - 116, height: canvas.height - 82 },
    state.flow.modes[mode],
    state.flow.grid,
    timeIndex,
    showArrows,
    mode === "conditional_moons" ? "condition 0: moons" : "condition 1: ring",
  );
  byId("conditional-time-value").textContent =
    state.flow.times[timeIndex].toFixed(3);

  const comparison = byId("conditional-comparison");
  const comparisonContext = clearCanvas(comparison);
  const gap = 28;
  const width = (comparison.width - gap * 3) / 2;
  [
    ["conditional_moons", "condition 0 → moons"],
    ["conditional_ring", "condition 1 → ring"],
  ].forEach(([key, title], panel) => {
    drawFlowContent(
      comparisonContext,
      {
        x: gap + panel * (width + gap),
        y: 34,
        width,
        height: comparison.height - 64,
      },
      state.flow.modes[key],
      state.flow.grid,
      timeIndex,
      false,
      title,
    );
  });
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

function drawSpriteCell(context, image, timeIndex, label, x, y, size) {
  context.imageSmoothingEnabled = false;
  context.drawImage(
    image,
    timeIndex * 28,
    label * 28,
    28,
    28,
    x,
    y,
    size,
    size,
  );
}

function drawMnistConditionGrid() {
  if (!state.predictions) return;
  const model = byId("condition-grid-model").value;
  const image = state.conditionGrids[model];
  const canvas = byId("mnist-condition-grid");
  const context = clearCanvas(canvas, "#111111");
  const left = 145;
  const top = 52;
  const cell = 64;
  const gap = 5;
  context.fillStyle = "#eeeeee";
  context.font = "bold 14px Arial";
  context.textAlign = "center";
  for (let sample = 0; sample < 4; sample += 1) {
    context.fillText(
      `noise ${sample + 1}`,
      left + sample * (cell + gap) + cell / 2,
      27,
    );
  }
  for (let label = 0; label < 10; label += 1) {
    context.fillStyle = "#eeeeee";
    context.font = "bold 15px Arial";
    context.textAlign = "right";
    context.fillText(
      `Label ${label}`,
      left - 18,
      top + label * (cell + gap) + cell / 2 + 5,
    );
    for (let sample = 0; sample < 4; sample += 1) {
      const x = left + sample * (cell + gap);
      const y = top + label * (cell + gap);
      context.imageSmoothingEnabled = false;
      context.drawImage(
        image,
        sample * 28,
        label * 28,
        28,
        28,
        x,
        y,
        cell,
        cell,
      );
    }
  }
  context.fillStyle = "#bbbbbb";
  context.font = "12px Arial";
  context.textAlign = "left";
  context.fillText(
    `${MODEL_LABELS[model]}：每个Label使用4个不同初始噪声`,
    left + 4 * (cell + gap) + 28,
    top + 22,
  );
}

function drawMnistComparison() {
  if (!state.predictions) return;
  const label = Number(byId("comparison-label").value);
  const canvas = byId("mnist-comparison");
  const context = clearCanvas(canvas, "#111111");
  const modelNames = ["additive", "adagn", "latent", "unet"];
  const count = state.predictions.times.length;
  const selected = Array.from({ length: 9 }, (_, index) =>
    Math.round((index / 8) * (count - 1)),
  );
  const left = 155;
  const top = 30;
  const rowHeight = (canvas.height - 50) / modelNames.length;
  const cellWidth = (canvas.width - left - 20) / selected.length;
  context.fillStyle = "#eeeeee";
  context.font = "bold 14px Arial";
  context.textAlign = "right";
  modelNames.forEach((model, row) => {
    context.fillText(
      MODEL_LABELS[model],
      left - 14,
      top + row * rowHeight + rowHeight / 2 + 5,
    );
    selected.forEach((timeIndex, column) => {
      const size = Math.min(rowHeight - 18, cellWidth - 8);
      const x = left + column * cellWidth + (cellWidth - size) / 2;
      const y = top + row * rowHeight + (rowHeight - size) / 2;
      drawSpriteCell(
        context,
        state.sprites[model],
        timeIndex,
        label,
        x,
        y,
        size,
      );
      if (row === 0) {
        context.fillStyle = "#bbbbbb";
        context.font = "10px Arial";
        context.textAlign = "center";
        context.fillText(
          state.predictions.times[timeIndex].toFixed(2),
          x + size / 2,
          17,
        );
        context.textAlign = "right";
        context.fillStyle = "#eeeeee";
        context.font = "bold 14px Arial";
      }
    });
  });
}

function drawMnist() {
  if (!state.predictions) return;
  const model = byId("mnist-model").value;
  const label = Number(byId("mnist-label").value);
  const timeIndex = Number(byId("mnist-time").value);
  const image = state.sprites[model];
  const canvas = byId("mnist-canvas");
  const context = clearCanvas(canvas, "#000000");
  drawSpriteCell(context, image, timeIndex, label, 0, 0, canvas.width);

  const timeline = byId("mnist-timeline");
  const timelineContext = clearCanvas(timeline, "#000000");
  const count = state.predictions.times.length;
  const cellWidth = timeline.width / count;
  for (let index = 0; index < count; index += 1) {
    const size = Math.max(8, Math.min(cellWidth - 1, 70));
    const x = index * cellWidth + (cellWidth - size) / 2;
    const y = (timeline.height - size) / 2;
    drawSpriteCell(timelineContext, image, index, label, x, y, size);
    if (index === timeIndex) {
      timelineContext.strokeStyle = "#ffffff";
      timelineContext.lineWidth = 2;
      timelineContext.strokeRect(x - 1, y - 1, size + 2, size + 2);
    }
  }

  const record = state.predictions.models[model][timeIndex];
  const confidence = record.confidence[label];
  const predicted = confidence >= 0.8 ? record.predicted[label] : "unknown";
  byId("mnist-time-value").textContent =
    state.predictions.times[timeIndex].toFixed(3);
  byId("mnist-target").textContent = String(label);
  byId("mnist-prediction").textContent = String(predicted);
  byId("mnist-confidence").textContent = `${(confidence * 100).toFixed(1)}%`;

  const descriptions = {
    additive: "条件只在入口相加，随后容易被归一化与局部卷积削弱。",
    adagn: "逐层AdaGN增强条件，但平坦CNN仍缺乏全局拓扑规划。",
    latent: "VAE压缩与Decoder先验减少像素自由度，但latent主干仍较简单。",
    unet: "Bottleneck、Skip、逐层条件与CFG共同建立稳定的类别运输。",
  };
  byId("mnist-explanation").textContent = descriptions[model];
}

function drawRealClusters(context, mapper, points, labels, alpha = 0.3) {
  const centroids = Array.from({ length: 10 }, () => [0, 0, 0]);
  points.forEach((point, index) => {
    const label = labels[index];
    const [x, y] = mapper(point);
    context.globalAlpha = alpha;
    context.fillStyle = COLORS[label];
    context.beginPath();
    context.arc(x, y, 2.7, 0, Math.PI * 2);
    context.fill();
    context.globalAlpha = 1;
    centroids[label][0] += point[0];
    centroids[label][1] += point[1];
    centroids[label][2] += 1;
  });
  context.font = "bold 13px Arial";
  context.textAlign = "center";
  centroids.forEach((centroid, label) => {
    const point = [centroid[0] / centroid[2], centroid[1] / centroid[2]];
    const [x, y] = mapper(point);
    context.fillStyle = COLORS[label];
    context.fillText(String(label), x, y);
  });
}

function getManifoldExtent(reduction) {
  const collections = [reduction.real.points];
  Object.values(reduction.models).forEach((model) => {
    model.forEach((time) => collections.push(time));
  });
  return paddedExtent(collections, 0.06);
}

function drawManifold() {
  if (!state.manifold) return;
  const reducer = byId("manifold-reducer").value;
  const reduction = state.manifold.reducers[reducer];
  const model = byId("manifold-model").value;
  const label = Number(byId("manifold-label").value);
  const timeIndex = Number(byId("manifold-time").value);
  const canvas = byId("manifold-canvas");
  const context = clearCanvas(canvas);
  const extent = getManifoldExtent(reduction);
  const mapper = createMapper(canvas, extent, 62);
  drawAxes(
    context,
    canvas,
    extent,
    mapper,
    `${reducer.toUpperCase()} 1`,
    `${reducer.toUpperCase()} 2`,
  );
  drawRealClusters(
    context,
    mapper,
    reduction.real.points,
    reduction.real.labels,
    0.32,
  );

  const expected = state.manifold.expected_labels;
  const modelPoints = reduction.models[model];
  const selected = expected
    .map((value, index) => (value === label ? index : -1))
    .filter((index) => index >= 0);

  selected.forEach((sample, trackIndex) => {
    context.strokeStyle = COLORS[label];
    context.globalAlpha =
      0.3 + (trackIndex / Math.max(selected.length - 1, 1)) * 0.42;
    context.lineWidth = 1.2;
    context.beginPath();
    for (let time = 0; time <= timeIndex; time += 1) {
      const [x, y] = mapper(modelPoints[time][sample]);
      if (time === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
    const start = mapper(modelPoints[0][sample]);
    context.fillStyle = COLORS[label];
    context.beginPath();
    context.arc(start[0], start[1], 2.7, 0, Math.PI * 2);
    context.fill();
  });
  context.globalAlpha = 1;

  selected.forEach((sample) => {
    const [x, y] = mapper(modelPoints[timeIndex][sample]);
    context.fillStyle = COLORS[label];
    context.beginPath();
    context.arc(x, y, 5.2, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "#ffffff";
    context.lineWidth = 1.2;
    context.stroke();
  });
  byId("manifold-time-value").textContent =
    state.manifold.times[timeIndex].toFixed(3);
}

function labelCentroid(points, expectedLabels, label) {
  const centroid = [0, 0];
  let count = 0;
  points.forEach((point, index) => {
    if (expectedLabels[index] !== label) return;
    centroid[0] += point[0];
    centroid[1] += point[1];
    count += 1;
  });
  return [centroid[0] / count, centroid[1] / count];
}

function getVelocityExtent(reducer) {
  if (reducer === "pca") {
    const collections = [state.velocity.real.points];
    Object.values(state.velocity.models).forEach((model) => {
      model.centroids.forEach((time) => collections.push(time));
    });
    return paddedExtent(collections, 0.08);
  }
  const reduction = state.manifold.reducers[reducer];
  const collections = [reduction.real.points];
  Object.values(reduction.models).forEach((model) => {
    model.forEach((time) => collections.push(time));
  });
  return paddedExtent(collections, 0.08);
}

function drawVelocity() {
  if (!state.velocity || !state.manifold) return;
  const reducer = byId("velocity-reducer").value;
  const model = byId("velocity-model").value;
  const timeIndex = Number(byId("velocity-time").value);
  const canvas = byId("velocity-canvas");
  const context = clearCanvas(canvas);
  const extent = getVelocityExtent(reducer);
  const mapper = createMapper(canvas, extent, 62);
  const axisPrefix =
    reducer === "pca" ? "classifier feature PCA" : reducer.toUpperCase();
  drawAxes(
    context,
    canvas,
    extent,
    mapper,
    `${axisPrefix} 1`,
    `${axisPrefix} 2`,
  );

  let realPoints;
  let realLabels;
  if (reducer === "pca") {
    realPoints = state.velocity.real.points;
    realLabels = state.velocity.real.labels;
  } else {
    realPoints = state.manifold.reducers[reducer].real.points;
    realLabels = state.manifold.reducers[reducer].real.labels;
  }
  drawRealClusters(context, mapper, realPoints, realLabels, 0.28);

  for (let label = 0; label < 10; label += 1) {
    let point;
    let change;
    if (reducer === "pca") {
      const data = state.velocity.models[model];
      point = data.centroids[timeIndex][label];
      const velocity = data.velocity[timeIndex][label];
      const visualHorizon = 0.12;
      change = [velocity[0] * visualHorizon, velocity[1] * visualHorizon];
    } else {
      const reduction = state.manifold.reducers[reducer];
      const points = reduction.models[model];
      point = labelCentroid(
        points[timeIndex],
        state.manifold.expected_labels,
        label,
      );
      const next = labelCentroid(
        points[timeIndex + 1],
        state.manifold.expected_labels,
        label,
      );
      const visualScale = reducer === "umap" ? 2.0 : 1.65;
      change = [
        (next[0] - point[0]) * visualScale,
        (next[1] - point[1]) * visualScale,
      ];
    }
    const start = mapper(point);
    const end = mapper([point[0] + change[0], point[1] + change[1]]);
    context.fillStyle = COLORS[label];
    context.beginPath();
    context.arc(start[0], start[1], 4.2, 0, Math.PI * 2);
    context.fill();
    drawArrow(
      context,
      start,
      [end[0] - start[0], end[1] - start[1]],
      COLORS[label],
      8,
      2.6,
    );
    context.fillStyle = COLORS[label];
    context.font = "bold 12px Arial";
    context.fillText(String(label), end[0] + 8, end[1] - 7);
  }
  byId("velocity-time-value").textContent =
    `${state.velocity.times[timeIndex].toFixed(3)} → ` +
    state.velocity.times[timeIndex + 1].toFixed(3);
  const notes = {
    pca: "PCA为线性映射：箭头是128维语义速度在同一线性基底中的投影，可解释方向与相对变化。",
    umap: "UMAP为非线性映射：箭头表示相邻时间点的UMAP嵌入位移，用于观察流形邻域变化，不作为严格速度向量。",
    tsne: "t-SNE为非线性局部投影：箭头表示相邻嵌入位移，可能出现投影放大或方向扭曲。",
  };
  byId("velocity-reducer-note").textContent = notes[reducer];
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
      : state.velocity.times
          .slice(0, -1)
          .map((value, index) => (value + state.velocity.times[index + 1]) / 2);
  const allValues = valuesByModel.flat();
  let minY = Math.min(...allValues);
  let maxY = Math.max(...allValues);
  if (metric === "alignment") {
    minY = Math.min(minY, 0);
    maxY = Math.max(maxY, 0);
  }
  const dy = Math.max(maxY - minY, 0.1);
  const extent = [0, 1, minY - dy * 0.08, maxY + dy * 0.08];
  const mapper = createMapper(canvas, extent, 60);
  const axisLabels = {
    speed: "mean semantic speed ||df/dt||",
    alignment: "cosine alignment",
    distance: "distance to target centroid",
  };
  drawAxes(context, canvas, extent, mapper, "Flow time t", axisLabels[metric]);
  if (metric === "alignment") {
    const start = mapper([0, 0]);
    const end = mapper([1, 0]);
    context.strokeStyle = "#777";
    context.beginPath();
    context.moveTo(...start);
    context.lineTo(...end);
    context.stroke();
  }
  names.forEach((name, modelIndex) => {
    const color = ["#2878b5", "#e6862f", "#3c8d85", "#b3242c"][modelIndex];
    const values = valuesByModel[modelIndex];
    context.strokeStyle = color;
    context.lineWidth = 2.2;
    context.beginPath();
    values.forEach((value, index) => {
      const [x, y] = mapper([times[index], value]);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
    context.fillStyle = color;
    context.font = "11px Arial";
    context.textAlign = "left";
    context.fillText(MODEL_LABELS[name], 72 + modelIndex * 196, 22);
  });
  const explanations = {
    speed:
      "速度大不代表方向正确。应与目标对齐度结合：U-Net在生成中期既移动快，也朝目标类别流形前进。",
    alignment:
      "大于0表示速度朝真实目标类别中心移动；后期变负可能表示已进入正确类别后转向具体书写风格。",
    distance:
      "距离下降表示进入目标语义区域；若准确率保持高而距离略回升，通常是在类别流形内部形成多样性。",
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
  ["unconditional-time", "unconditional-arrows"].forEach((id) =>
    byId(id).addEventListener("input", drawUnconditional),
  );
  ["conditional-mode", "conditional-time", "conditional-arrows"].forEach((id) =>
    byId(id).addEventListener("input", drawConditional),
  );
  byId("condition-grid-model").addEventListener(
    "input",
    drawMnistConditionGrid,
  );
  byId("comparison-label").addEventListener("input", drawMnistComparison);
  ["mnist-model", "mnist-label", "mnist-time"].forEach((id) =>
    byId(id).addEventListener("input", drawMnist),
  );
  [
    "manifold-reducer",
    "manifold-model",
    "manifold-label",
    "manifold-time",
  ].forEach((id) => byId(id).addEventListener("input", drawManifold));
  ["velocity-reducer", "velocity-model", "velocity-time"].forEach((id) =>
    byId(id).addEventListener("input", drawVelocity),
  );
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
      additiveConditions,
      adagnConditions,
      latentConditions,
      unetConditions,
    ] = await Promise.all([
      loadJson("assets/flow_2d.json"),
      loadJson("assets/mnist_predictions.json"),
      loadJson("assets/semantic_reductions.json"),
      loadJson("assets/semantic_velocity.json"),
      loadImage("assets/mnist_additive_sprites.png"),
      loadImage("assets/mnist_adagn_sprites.png"),
      loadImage("assets/mnist_latent_sprites.png"),
      loadImage("assets/mnist_unet_sprites.png"),
      loadImage("assets/mnist_additive_conditions.png"),
      loadImage("assets/mnist_adagn_conditions.png"),
      loadImage("assets/mnist_latent_conditions.png"),
      loadImage("assets/mnist_unet_conditions.png"),
    ]);
    state.flow = flow;
    state.predictions = predictions;
    state.manifold = manifold;
    state.velocity = velocity;
    state.sprites = { additive, adagn, latent, unet };
    state.conditionGrids = {
      additive: additiveConditions,
      adagn: adagnConditions,
      latent: latentConditions,
      unet: unetConditions,
    };

    populateNumericSelect(byId("comparison-label"), 10);
    populateNumericSelect(byId("mnist-label"), 10);
    populateNumericSelect(byId("manifold-label"), 10);
    populateModelSelect(
      byId("manifold-model"),
      Object.keys(manifold.reducers.pca.models),
    );
    populateModelSelect(byId("velocity-model"), Object.keys(velocity.models));
    byId("manifold-model").value = "conditional_unet";
    byId("velocity-model").value = "conditional_unet";

    const flowMax = flow.times.length - 1;
    byId("unconditional-time").max = String(flowMax);
    byId("conditional-time").max = String(flowMax);
    const mnistMax = predictions.times.length - 1;
    byId("mnist-time").max = String(mnistMax);
    byId("manifold-time").max = String(manifold.times.length - 1);
    byId("velocity-time").max = String(velocity.times.length - 2);

    attachInteractions();
    drawUnconditional();
    drawConditional();
    drawMnistConditionGrid();
    drawMnistComparison();
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
