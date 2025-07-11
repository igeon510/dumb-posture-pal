const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  onPostureAlert: (callback) => ipcRenderer.on('posture-alert', callback),
  resetBaseline: () => ipcRenderer.send('reset-baseline'),
  quitApp: () => ipcRenderer.send('quit-app'),
  setIgnoreMouseEvents: (ignore) => ipcRenderer.send('set-ignore-mouse-events', ignore),
  animationFinished: () => ipcRenderer.send('animation-finished'),
  onInitialMessage: (callback) => ipcRenderer.on('initial-message', callback),
  onBaselineSetMessage: (callback) => ipcRenderer.on('baseline-set-message', callback)
});
