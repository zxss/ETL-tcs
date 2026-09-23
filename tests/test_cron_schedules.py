"""Расписания cron: каталог лога обязан создаваться той же строкой.

23.09.2026 весь турнирный контур простоял сутки: в crontab стояло
`>> audit/stage2-tournament/cron.log`, а каталога не было. Перенаправление bash
выполняет ДО запуска python — цепочка падала на `No such file or directory`,
лога не появлялось, заявок не было, и снаружи это выглядело как «всё тихо».
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
FILES = ("stage2_crontab", "stage2_tournament_crontab", "stage2_both_staggered")

_CRON_LINE = re.compile(r"^[\d*][^ \t]*(?:[ \t]+[^ \t]+){4}[ \t]+(?P<cmd>.+)$")
_REDIRECT = re.compile(r">>\s*(?P<target>\S+)")


def _command_lines(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" in line.split()[0]:
            continue
        m = _CRON_LINE.match(line)
        if m:
            out.append(m.group("cmd"))
    return out


class TestCronSchedules(unittest.TestCase):

    def test_files_exist(self):
        for name in FILES:
            self.assertTrue((SCRIPTS / name).is_file(), f"нет {name}")

    def test_every_redirect_dir_is_created(self):
        """У каждой строки каталог лога создаётся в той же цепочке."""
        for name in FILES:
            text = (SCRIPTS / name).read_text(encoding="utf-8")
            cmds = _command_lines(text)
            self.assertTrue(cmds, f"{name}: не разобрана ни одна строка")
            for cmd in cmds:
                m = _REDIRECT.search(cmd)
                self.assertIsNotNone(m, f"{name}: строка без >> лога: {cmd}")
                target = m.group("target")
                if target.startswith("$"):          # LOG=<путь> из шапки
                    target = re.search(rf"^{re.escape(target[1:])}=(\S+)",
                                       text, re.M).group(1)
                log_dir = target.rsplit("/", 1)[0]
                self.assertIn(f"mkdir -p {log_dir}", cmd.replace("$DIR", log_dir),
                              f"{name}: каталог {log_dir} не создаётся строкой: {cmd}")

    def test_tournament_lines_carry_prod_flag_and_profile(self):
        text = (SCRIPTS / "stage2_tournament_crontab").read_text(encoding="utf-8")
        for cmd in _command_lines(text):
            self.assertIn("--prod", cmd)
            self.assertIn(".env.tournament", cmd)

    def test_staggered_profiles_never_start_together(self):
        """Профили делят реестр стопов: одновременный старт теряет прогон.

        Совпадения ВНУТРИ профиля (protect каждые 5 минут накладывается на
        фазу) штатны — protect берёт блокировку с wait=False и пропускает
        прогон. Недопустимо пересечение песочницы и турнира между собой.
        """
        text = (SCRIPTS / "stage2_both_staggered").read_text(encoding="utf-8")
        slots: dict[str, set[tuple[int, int]]] = {"sandbox": set(), "prod": set()}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" in line.split()[0]:
                continue
            if not _CRON_LINE.match(line):
                continue
            fields = line.split()
            profile = "prod" if "--prod" in line else "sandbox"
            for mi in _expand(fields[0], 0, 59):
                for ho in _expand(fields[1], 0, 23):
                    slots[profile].add((ho, mi))
        clash = sorted(slots["sandbox"] & slots["prod"])
        self.assertFalse(clash, f"песочница и турнир стартуют вместе: {clash[:5]}")


def _expand(field: str, lo: int, hi: int) -> list[int]:
    out: list[int] = []
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/")
            step = int(s)
        if part == "*":
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-")
            start, end = int(a), int(b)
        else:
            start = end = int(part)
        out.extend(range(start, end + 1, step))
    return out


if __name__ == "__main__":
    unittest.main()
