# Original author: Tawn Kramer

import asyncio
import json
import re


def replace_float_notation(string):
    """
    Replace unity float notation for languages like
    French or German that use comma instead of dot.
    This convert the json sent by Unity to a valid one.
    Ex: "test": 1,2, "key": 2 -> "test": 1.2, "key": 2

    :param string: (str) The incorrect json string
    :return: (str) Valid JSON string
    """
    regex_french_notation = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+),'
    regex_end = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+)}'

    for regex in [regex_french_notation, regex_end]:
        matches = re.finditer(regex, string, re.MULTILINE)

        for match in matches:
            num = match.group('num').replace(',', '.')
            string = string.replace(match.group('num'), num)
    return string


class IMesgHandler(object):
    """
    Abstract class that represent a socket message handler.
    """
    def on_connect(self, socket_handler):
        pass

    def on_recv_message(self, message):
        pass

    def on_close(self):
        pass

    def on_disconnect(self):
        pass


class SimServer:
    """
    Receives network connections and establishes handlers for each client.
    Each client connection is handled by a new instance of the SimHandler class.

    :param address: (str, int) (address, port)
    :param msg_handler: (socket message handler object)
    """
    def __init__(self, address, msg_handler):
        self.address = address
        self.msg_handler = msg_handler
        self.sim_handler = None
        self.server = None

    async def start(self):
        """Start the server"""
        self.server = await asyncio.start_server(
            self.handle_client, self.address[0], self.address[1]
        )
        addr = self.server.sockets[0].getsockname()
        print(f'Binding to {addr}')

    async def handle_client(self, reader, writer):
        """Called when a client connects"""
        addr = writer.get_extra_info('peername')
        print('Got a new client', addr)
        
        self.sim_handler = SimHandler(reader, writer, self.msg_handler)
        await self.sim_handler.run()

    async def serve_forever(self):
        """Keep the server running"""
        async with self.server:
            await self.server.serve_forever()

    def close(self):
        """Close the server"""
        print("Server shutdown")
        if self.server:
            self.server.close()
        
        if self.msg_handler is not None:
            self.msg_handler.on_close()

        if self.sim_handler is not None:
            self.sim_handler.close()


class SimHandler:
    """
    Handles messages from a single TCP client.

    :param reader: (asyncio.StreamReader)
    :param writer: (asyncio.StreamWriter)
    :param msg_handler: (socket message handler object)
    :param chunk_size: (int)
    """

    def __init__(self, reader, writer, msg_handler=None, chunk_size=(16 * 1024)):
        self.reader = reader
        self.writer = writer
        self.msg_handler = msg_handler
        self.chunk_size = chunk_size
        self.data_to_read = []

        if msg_handler:
            msg_handler.on_connect(self)

    def queue_message(self, msg):
        """Queue a message to send to the client"""
        json_msg = json.dumps(msg)
        self.writer.write((json_msg + '\n').encode())

    async def run(self):
        """Main loop for handling client communication"""
        try:
            while True:
                data = await self.reader.read(self.chunk_size)
                
                if len(data) == 0:
                    # Connection closed
                    break

                self.data_to_read.append(data.decode("utf-8"))

                messages = ''.join(self.data_to_read).split('\n')
                self.data_to_read = []

                for mesg in messages:
                    if len(mesg) < 2:
                        continue
                    if mesg[0] == '{' and mesg[-1] == '}':
                        self.handle_json_message(mesg)
                    else:
                        self.data_to_read.append(mesg)

                # Flush any queued messages
                await self.writer.drain()

        except Exception as e:
            print(f"Error in client handler: {e}")
        finally:
            self.close()

    def handle_json_message(self, chunk):
        """
        We are expecting a json object.
        
        :param chunk: (str)
        """
        try:
            chunk = replace_float_notation(chunk)
            json_obj = json.loads(chunk)
        except Exception as e:
            print(e, 'failed to read json ', chunk)
            return

        try:
            if self.msg_handler:
                self.msg_handler.on_recv_message(json_obj)
        except Exception as e:
            print(e, '>>> failure during on_recv_message:', chunk)

    def close(self):
        """Close the connection"""
        if self.msg_handler is not None:
            self.msg_handler.on_disconnect()
            self.msg_handler = None
            print('Connection dropped')

        self.writer.close()
        # Await writer.wait_closed() only in Python 3.7+
        asyncio.create_task(self.writer.wait_closed())
        self.reader = None
        self.writer = None
        print('Connection closed')