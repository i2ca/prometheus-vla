"""Fail before starting another recorder on an existing session's ports."""
import socket


def require_free_ports(ports):
    sockets = []
    try:
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError as exc:
                raise RuntimeError(
                    f"Porta {port} indisponível: encerre a sessão anterior antes de "
                    "gravar novamente. Nenhum processo foi encerrado automaticamente. "
                    "Depois de iniciar a nova sessão, atualize o navegador do Quest "
                    "e entre no modo VR."
                ) from exc
    finally:
        for sock in sockets:
            sock.close()
