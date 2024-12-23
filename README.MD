# Lab Simulation

這是一個模擬氣體粒子在三維空間中運動和反應的 Python 程式。該程式使用 Matplotlib 進行三維可視化和動畫，並記錄溫度和 CO₂ 濃度的變化。

## 目錄

- [安裝](#安裝)
- [使用方法](#使用方法)
- [程式說明](#程式說明)
- [輸出](#輸出)

## 安裝

1. 確保您已經安裝了 Python 3.6 或更高版本。
2. 創建並激活虛擬環境：

    ```sh
    python -m venv myenv
    source myenv/bin/activate  # 對於 Windows 使用 myenv\Scripts\activate
    ```

3. 安裝所需的 Python 套件：

    ```sh
    pip install numpy matplotlib pandas
    ```

4. 安裝 ffmpeg 以支持動畫保存：

    - 對於 macOS，用 Homebrew 安裝：

        ```sh
        brew install ffmpeg
        ```

    - 對於 Windows，請從 [ffmpeg 官網](https://ffmpeg.org/download.html) 下載並安裝。

## 使用方法

1. 將 [`lab.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "/Users/chenbaiyan/Desktop/of/lab.py") 文件放在您的工作目錄中。
2. 在終端中運行 [`lab.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "/Users/chenbaiyan/Desktop/of/lab.py")：

    ```sh
    python -u "/path/to/lab.py"
    ```

## 程式說明

### 系統參數

- [`num_particles`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A7%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A7%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 粒子數量
- [`time_steps`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A8%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A8%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 模擬時間步數
- [`box_size`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A9%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A9%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 模擬空間的尺寸
- [`dt`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A10%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A10%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 時間步長
- [`T0`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A11%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A11%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 初始溫度
- [`alpha`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A12%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A12%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 反應放熱量
- [`center`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A13%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A13%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 熱源中心位置
- [`r`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A14%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A14%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 反應半徑
- [`D0`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A15%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A15%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 擴散係數
- [`beta`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A16%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A16%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 擴散係數溫度依賴性
- [`gravity`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A17%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A17%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 重力加速度
- [`reaction_rate`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A18%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A18%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 化學反應速率

### 氣體參數

- [`thermal_conductivity`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A22%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A22%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 熱導率
- [`specific_heat`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A23%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A23%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 比熱容
- [`activation_energy`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A24%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A24%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 活化能
- [`R`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A25%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A25%2C%22character%22%3A0%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 氣體常數

### 數據記錄器

[`DataRecorder`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A28%2C%22character%22%3A6%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A28%2C%22character%22%3A6%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition") 類用於記錄模擬過程中的溫度和 CO₂ 濃度變化。

### 函數

- [`initialize_particles(num_particles, box_size)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A63%2C%22character%22%3A4%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 初始化粒子位置、速度、溫度和類型。
- [`heat_conduction(temperatures, positions, thermal_conductivity, dt)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A71%2C%22character%22%3A4%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 計算熱傳導。
- [`update_particles_with_combustion(positions, velocities, temperatures, particle_types, dt, box_size, reaction_rate)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A82%2C%22character%22%3A4%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 更新粒子狀態，包括化學反應和物理運動。
- [`calculate_co2_density(positions, particle_types, y_plane, box_size, resolution)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A237%2C%22character%22%3A4%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 計算特定 y 截面的 CO₂ 濃度。
- [`plot_co2_density(density, z_plane, box_size, filename)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A257%2C%22character%22%3A4%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 繪製並儲存特定 z 截面的 CO₂ 濃度圖。
- [`save_concentration_data(recorder, filename)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A267%2C%22character%22%3A4%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 將濃度數據保存到 CSV 文件。
- [`save_concentration_plot(recorder, filename)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A280%2C%22character%22%3A4%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 創建並保存高質量的濃度圖。
- [`animate(frame)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A315%2C%22character%22%3A4%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 更新動畫每一幀的粒子狀態和圖形。
- [`save_simulation_data(recorder)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Flab.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A407%2C%22character%22%3A4%7D%7D%5D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "Go to definition"): 保存模擬數據和圖表。

## 輸出

- [`simulation_animation.mp4`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Fsimulation_animation.mp4%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "/Users/chenbaiyan/Desktop/of/simulation_animation.mp4"): 模擬動畫。
- `co2_density_z10.png`: z=10 截面的 CO₂ 濃度圖。
- [`concentration_data.csv`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Fconcentration_data.csv%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "/Users/chenbaiyan/Desktop/of/concentration_data.csv"): 濃度數據 CSV 文件。
- [`concentration_plot.png`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fchenbaiyan%2FDesktop%2Fof%2Fconcentration_plot.png%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%228cb34413-9823-47d2-be01-007e8688e5a1%22%5D "/Users/chenbaiyan/Desktop/of/concentration_plot.png"): 濃度變化圖。

## 注意事項

- 請確保安裝了所有所需的 Python 套件和 ffmpeg。
- 根據需要調整系統參數和氣體參數以適應不同的模擬需求。