"""
Modern dark-themed stylesheet inspired by mission-critical aerospace and robotics ground stations.
High contrast, clean typography, unambiguous color badges for hardware status.
"""

DARK_THEME_QSS = """
/* Global Application Style */
QWidget {
    background-color: #121417;
    color: #E0E6ED;
    font-family: "Segoe UI", "Roboto", "Ubuntu", sans-serif;
    font-size: 13px;
    selection-background-color: #007ACC;
    selection-color: #FFFFFF;
}

/* Tooltips */
QToolTip {
    background-color: #1E232A;
    color: #F0F4F8;
    border: 1px solid #3A4454;
    padding: 5px;
    border-radius: 4px;
}

/* Panels and Cards */
QFrame.CardPanel {
    background-color: #181C22;
    border: 1px solid #28303C;
    border-radius: 6px;
    padding: 10px;
}

QGroupBox {
    background-color: #181C22;
    border: 1px solid #28303C;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 16px;
    font-weight: bold;
    font-size: 12px;
    color: #7E92A8;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: #63B3ED;
}

/* Header Bar */
QFrame#HeaderBar {
    background-color: #161A20;
    border-bottom: 2px solid #242B36;
    min-height: 48px;
    max-height: 48px;
    padding: 0 16px;
}

QLabel#AppTitle {
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #E2E8F0;
}

QLabel#AppSubtitle {
    font-size: 11px;
    font-weight: normal;
    color: #718096;
}

/* Status Badges */
QLabel.StatusBadge {
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 11px;
}

QLabel.StatusActive {
    background-color: #143826;
    color: #48BB78;
    border: 1px solid #2F855A;
}

QLabel.StatusInactive {
    background-color: #2D3748;
    color: #A0AEC0;
    border: 1px solid #4A5568;
}

QLabel.StatusWarning {
    background-color: #3D2D14;
    color: #ED8936;
    border: 1px solid #C05621;
}

QLabel.StatusError {
    background-color: #3B1818;
    color: #F56565;
    border: 1px solid #E53E3E;
}

/* Metrics Displays */
QLabel.MetricValue {
    font-size: 22px;
    font-weight: bold;
    color: #FFFFFF;
}

QLabel.MetricValueHighlight {
    font-size: 22px;
    font-weight: bold;
    color: #48BB78;
}

QLabel.MetricUnit {
    font-size: 11px;
    color: #718096;
}

QLabel.MetricLabel {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #A0AEC0;
}

/* Buttons */
QPushButton {
    background-color: #232934;
    color: #E2E8F0;
    border: 1px solid #364152;
    border-radius: 5px;
    padding: 8px 14px;
    font-weight: 600;
    font-size: 12px;
    min-height: 18px;
}

QPushButton:hover {
    background-color: #2D3543;
    border-color: #4B5A72;
}

QPushButton:pressed {
    background-color: #1A202A;
}

QPushButton:disabled {
    background-color: #161A20;
    color: #4A5568;
    border-color: #242B36;
}

QPushButton.PrimaryButton {
    background-color: #2B6CB0;
    border: 1px solid #3182CE;
    color: #FFFFFF;
}

QPushButton.PrimaryButton:hover {
    background-color: #3182CE;
}

QPushButton.SuccessButton {
    background-color: #276749;
    border: 1px solid #2F855A;
    color: #FFFFFF;
}

QPushButton.SuccessButton:hover {
    background-color: #2F855A;
}

QPushButton.DangerButton {
    background-color: #9B2C2C;
    border: 1px solid #C53030;
    color: #FFFFFF;
}

QPushButton.DangerButton:hover {
    background-color: #C53030;
}

/* Tables */
QTableWidget {
    background-color: #15181E;
    border: 1px solid #242B36;
    gridline-color: #1E232B;
    border-radius: 4px;
}

QTableWidget::item {
    padding: 4px 8px;
}

QTableWidget::item:selected {
    background-color: #2C3E50;
}

QHeaderView::section {
    background-color: #1B2027;
    color: #A0AEC0;
    padding: 6px;
    border: 1px solid #242B36;
    font-weight: bold;
    font-size: 11px;
}

/* ScrollBars */
QScrollBar:vertical {
    background: #14171C;
    width: 8px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #2D3748;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #4A5568;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Inputs and Combos */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #1A1F26;
    border: 1px solid #2E3846;
    border-radius: 4px;
    padding: 6px 10px;
    color: #E2E8F0;
}

QLineEdit:focus, QComboBox:focus {
    border: 1px solid #3182CE;
}
"""

