const { app, BrowserWindow, ipcMain, screen, Tray, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

function createWindow() {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;

  const win = new BrowserWindow({
    width: width,
    height: height,
    x: 0,
    y: 0,
    transparent: true,
    frame: false,
    hasShadow: false, // 창 그림자 제거
    alwaysOnTop: true,
    show: false, // 기본적으로 창 숨김
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
    },
  });

  win.loadFile('index.html');
  win.setIgnoreMouseEvents(true); // 기본적으로 마우스 이벤트 무시
  // win.webContents.openDevTools(); // 개발자 도구 열기 (디버깅용)
}

let pythonProcess;

function startPythonProcess() {
  if (pythonProcess) {
    pythonProcess.kill();
  }

  pythonProcess = spawn('python3', [path.join(__dirname, 'posture_detector.py')]);

  pythonProcess.stdout.on('data', (data) => {
    const message = data.toString();
    console.log(`Python stdout: ${message}`);
    if (message.includes('POSTURE_ALERT')) {
      const allWindows = BrowserWindow.getAllWindows();
      if (allWindows.length > 0) {
        allWindows[0].show(); // 알림 시 창 보이기
        allWindows[0].webContents.send('posture-alert');
      }
    } else if (message.includes('안녕하세요! Posture Pal이 당신의 자세를 감지하기 시작합니다.')) {
      const allWindows = BrowserWindow.getAllWindows();
      if (allWindows.length > 0) {
        allWindows[0].show(); // 시작 메시지 시 창 보이기
        allWindows[0].webContents.send('initial-message');
      }
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python stderr: ${data.toString()}`);
  });
}

app.whenReady().then(() => {
  createWindow();
  startPythonProcess(); // 초기 실행

  // 트레이 아이콘 설정
  const iconPath = path.join(__dirname, 'iconTemplate.png'); // 아이콘 파일 경로
  const tray = new Tray(iconPath);

  function updateTrayMenu() {
    const contextMenu = Menu.buildFromTemplate([
      {
        label: 'Reset Baseline',
        click: () => {
          ipcMain.emit('reset-baseline');
          const allWindows = BrowserWindow.getAllWindows();
          if (allWindows.length > 0) {
            allWindows[0].show(); // 메시지 표시를 위해 창 보이기
            allWindows[0].webContents.send('baseline-set-message');
          }
        }
      },
      { type: 'separator' },
      {
        label: 'Quit',
        click: () => {
          ipcMain.emit('quit-app');
        }
      }
    ]);
    tray.setContextMenu(contextMenu);
  }

  updateTrayMenu(); // 초기 메뉴 설정

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// 신호 수신
ipcMain.on('reset-baseline', () => {
  console.log('Sending RESET to Python...');
  if (pythonProcess) {
    pythonProcess.stdin.write("RESET\n");
  }
});

ipcMain.on('quit-app', () => {
  app.quit();
});

// 앱 종료 직전에 Python 프로세스 종료
app.on('before-quit', () => {
  if (pythonProcess) {
    console.log('Killing Python process...');
    pythonProcess.kill();
  }
});

// 마우스 이벤트 처리 신호
ipcMain.on('set-ignore-mouse-events', (event, ignore) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  win.setIgnoreMouseEvents(ignore);
});

ipcMain.on('animation-finished', (event) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  win.hide(); // 애니메이션 종료 후 창 숨기기
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
