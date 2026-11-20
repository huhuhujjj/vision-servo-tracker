"use strict";

const canvas = document.querySelector("#view");
const context = canvas.getContext("2d");
const statusLine = document.querySelector("#status");
const message = document.querySelector("#message");
const token = document.querySelector('meta[name="control-token"]').content;
let displayedFrame = null;

function showState(state) {
  const labels = { idle: "请选择目标", tracking: "跟踪中", lost: "LOST · 目标丢失", offline: "画面中断" };
  let text = `${labels[state.status] || state.status}　${state.class_name || ""}`;
  if (state.track_id !== null && state.track_id !== undefined) text += ` #${state.track_id}`;
  text += `　${state.fps || 0} 帧/秒`;
  if (state.offset_px) text += `　dx=${state.offset_px[0].toFixed(0)}　dy=${state.offset_px[1].toFixed(0)}`;
  statusLine.textContent = text;
}

async function command(data) {
  try {
    const response = await fetch("/command", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Control-Token": token },
      body: JSON.stringify(data),
      signal: AbortSignal.timeout(5000),
    });
    const result = await response.json();
    message.textContent = response.ok ? "操作已接收" : result.error;
  } catch {
    message.textContent = "连接中断，请检查 SSH 窗口";
  }
}

canvas.addEventListener("click", event => {
  if (displayedFrame === null) return;
  const bounds = canvas.getBoundingClientRect();
  // 回传实际显示帧的编号，服务端不会拿另一帧的框来解释这次点击。
  command({ action: "select", frame_id: displayedFrame,
    x: (event.clientX - bounds.left) / bounds.width,
    y: (event.clientY - bounds.top) / bounds.height });
});
document.querySelector("#release").addEventListener("click", () => command({ action: "release" }));

async function refresh() {
  try {
    const response = await fetch("/frame", { cache: "no-store", signal: AbortSignal.timeout(6000) });
    if (!response.ok) {
      const state = await response.json();
      throw new Error(state.reason || "正在加载模型");
    }
    // 图像和状态在服务端同一次加锁中读取，避免状态比视频快一帧。
    const state = JSON.parse(response.headers.get("X-Tracking-State"));
    const frameId = Number(response.headers.get("X-Frame-ID"));
    const bitmap = await createImageBitmap(await response.blob());
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    context.drawImage(bitmap, 0, 0);
    bitmap.close();
    if (state.status === "tracking" && state.box) {
      const [x1, y1, x2, y2] = state.box;
      context.strokeStyle = "#ffff00";
      context.lineWidth = 4;
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    displayedFrame = frameId;
    canvas.classList.remove("offline");
    showState(state);
  } catch (error) {
    displayedFrame = null;
    canvas.classList.add("offline");
    // 断线后覆盖状态行，避免继续展示失效的 dx/dy。
    statusLine.textContent = `画面不可用：${error.message}`;
  }
  // 串行请求；网络慢时不叠加多个预览请求。
  setTimeout(refresh, 80);
}
refresh();
