import QtQuick 2.0
import QtQuick.Controls 2.1
import QtQuick.Controls.Material 2.3

ApplicationWindow
{
    width: 640
    height: 480
    visible: true

    Material.theme: Material.Dark

    Button {
        id: button
        x: 150
        y: 222
        text: qsTr("Button")
    }
}
