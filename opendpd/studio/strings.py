"""Strings the desktop window shows outside the page: the quit question and pywebview's own dialogs and menus.

The page translates itself (frontend/src/i18n). These host-side strings follow
the same UI languages; the launcher picks them from the workspace setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ShellStrings:
    quit_title: str
    quit_body: str                                   # carries a {count} placeholder
    localization: Dict[str, str] = field(default_factory=dict)   # pywebview localization keys


LOCALIZATION_KEYS = (
    "global.quitConfirmation", "global.ok", "global.quit", "global.cancel", "global.saveFile",
    "cocoa.menu.about", "cocoa.menu.services", "cocoa.menu.view", "cocoa.menu.edit", "cocoa.menu.hide",
    "cocoa.menu.hideOthers", "cocoa.menu.showAll", "cocoa.menu.quit", "cocoa.menu.fullscreen",
    "cocoa.menu.cut", "cocoa.menu.copy", "cocoa.menu.paste", "cocoa.menu.selectAll",
    "windows.fileFilter.allFiles", "windows.fileFilter.otherFiles",
    "linux.openFile", "linux.openFiles", "linux.openFolder",
)


def _localization(*values: str) -> Dict[str, str]:
    assert len(values) == len(LOCALIZATION_KEYS)
    return dict(zip(LOCALIZATION_KEYS, values))


ENGLISH = ShellStrings(
    quit_title="Quit OpenDPD Studio?",
    quit_body="{count} experiment(s) are running. Quit OpenDPD Studio and stop them?",
    localization=_localization(
        "Do you really want to quit?", "OK", "Quit", "Cancel", "Save file",
        "About", "Services", "View", "Edit", "Hide", "Hide Others", "Show All", "Quit", "Enter Fullscreen",
        "Cut", "Copy", "Paste", "Select All", "All files", "Other file types",
        "Open file", "Open files", "Open folder"),
)

FRENCH = ShellStrings(
    quit_title="Quitter OpenDPD Studio ?",
    quit_body="{count} expérience(s) en cours. Quitter OpenDPD Studio et les arrêter ?",
    localization=_localization(
        "Voulez-vous vraiment quitter ?", "OK", "Quitter", "Annuler", "Enregistrer le fichier",
        "À propos", "Services", "Présentation", "Édition", "Masquer", "Masquer les autres", "Tout afficher",
        "Quitter", "Activer le mode plein écran", "Couper", "Copier", "Coller", "Tout sélectionner",
        "Tous les fichiers", "Autres types de fichiers", "Ouvrir un fichier", "Ouvrir des fichiers", "Ouvrir un dossier"),
)

GERMAN = ShellStrings(
    quit_title="OpenDPD Studio beenden?",
    quit_body="{count} Experiment(e) laufen noch. OpenDPD Studio beenden und sie abbrechen?",
    localization=_localization(
        "Möchten Sie wirklich beenden?", "OK", "Beenden", "Abbrechen", "Datei sichern",
        "Über", "Dienste", "Darstellung", "Bearbeiten", "Ausblenden", "Andere ausblenden", "Alle einblenden",
        "Beenden", "Vollbild aktivieren", "Ausschneiden", "Kopieren", "Einsetzen", "Alles auswählen",
        "Alle Dateien", "Andere Dateitypen", "Datei öffnen", "Dateien öffnen", "Ordner öffnen"),
)

SPANISH = ShellStrings(
    quit_title="¿Salir de OpenDPD Studio?",
    quit_body="{count} experimento(s) en ejecución. ¿Salir de OpenDPD Studio y detenerlos?",
    localization=_localization(
        "¿Seguro que desea salir?", "Aceptar", "Salir", "Cancelar", "Guardar archivo",
        "Acerca de", "Servicios", "Visualización", "Edición", "Ocultar", "Ocultar otros", "Mostrar todo",
        "Salir", "Usar pantalla completa", "Cortar", "Copiar", "Pegar", "Seleccionar todo",
        "Todos los archivos", "Otros tipos de archivo", "Abrir archivo", "Abrir archivos", "Abrir carpeta"),
)

CHINESE = ShellStrings(
    quit_title="退出 OpenDPD Studio？",
    quit_body="还有 {count} 个实验正在运行。要退出 OpenDPD Studio 并停止它们吗？",
    localization=_localization(
        "确定要退出吗？", "好", "退出", "取消", "保存文件",
        "关于", "服务", "显示", "编辑", "隐藏", "隐藏其他", "全部显示", "退出", "进入全屏幕",
        "剪切", "拷贝", "粘贴", "全选", "所有文件", "其他文件类型", "打开文件", "打开多个文件", "打开文件夹"),
)

JAPANESE = ShellStrings(
    quit_title="OpenDPD Studio を終了しますか？",
    quit_body="{count} 件の実験が実行中です。OpenDPD Studio を終了して停止しますか？",
    localization=_localization(
        "本当に終了しますか？", "OK", "終了", "キャンセル", "ファイルを保存",
        "について", "サービス", "表示", "編集", "隠す", "ほかを隠す", "すべてを表示", "終了", "フルスクリーンにする",
        "カット", "コピー", "ペースト", "すべてを選択", "すべてのファイル", "その他のファイル形式",
        "ファイルを開く", "複数のファイルを開く", "フォルダを開く"),
)

KOREAN = ShellStrings(
    quit_title="OpenDPD Studio를 종료할까요?",
    quit_body="{count}개의 실험이 실행 중입니다. OpenDPD Studio를 종료하고 중지할까요?",
    localization=_localization(
        "정말 종료하시겠습니까?", "확인", "종료", "취소", "파일 저장",
        "정보", "서비스", "보기", "편집", "가리기", "나머지 가리기", "모두 보기", "종료", "전체 화면 시작",
        "오려두기", "복사하기", "붙여넣기", "전체 선택", "모든 파일", "기타 파일 유형",
        "파일 열기", "여러 파일 열기", "폴더 열기"),
)

DUTCH = ShellStrings(
    quit_title="OpenDPD Studio afsluiten?",
    quit_body="Er lopen nog {count} experimenten. OpenDPD Studio afsluiten en deze stoppen?",
    localization=_localization(
        "Weet u zeker dat u wilt afsluiten?", "OK", "Sluit af", "Annuleer", "Bewaar bestand",
        "Over", "Voorzieningen", "Weergave", "Wijzig", "Verberg", "Verberg andere", "Toon alles",
        "Sluit af", "Schakel schermvullende weergave in", "Knip", "Kopieer", "Plak", "Selecteer alles",
        "Alle bestanden", "Andere bestandstypen", "Open bestand", "Open bestanden", "Open map"),
)

ITALIAN = ShellStrings(
    quit_title="Uscire da OpenDPD Studio?",
    quit_body="Ci sono {count} esperimenti in esecuzione. Uscire da OpenDPD Studio e interromperli?",
    localization=_localization(
        "Vuoi davvero uscire?", "OK", "Esci", "Annulla", "Salva file",
        "Informazioni", "Servizi", "Vista", "Modifica", "Nascondi", "Nascondi altre", "Mostra tutte",
        "Esci", "Attiva modalità a tutto schermo", "Taglia", "Copia", "Incolla", "Seleziona tutto",
        "Tutti i file", "Altri tipi di file", "Apri file", "Apri file", "Apri cartella"),
)

STRINGS: Dict[str, ShellStrings] = {
    "nl": DUTCH, "it": ITALIAN,
    "en": ENGLISH, "fr": FRENCH, "de": GERMAN, "es": SPANISH, "zh": CHINESE, "ja": JAPANESE, "ko": KOREAN,
}


def shell_strings(language: Optional[str] = None) -> ShellStrings:
    """Strings for a UI language code; English for an unknown or missing code."""
    return STRINGS.get(language or "", ENGLISH)
