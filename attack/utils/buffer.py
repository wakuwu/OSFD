import os
import os.path as osp
import mmcv
import torch
from mmcv.parallel import scatter


class Buffer:

    def __init__(self, buffer_dir) -> None:
        self.buffer_dir = buffer_dir
        # memory buffer
        self.buffer_dict = dict()
        # The buffer location corresponding to the buffer variable vname
        self.buffer_type_dict = dict()
        self.global_buffer_type = "memory"
        self.device = os.environ["device"]

    def update_buffer_type(self, vname, buffer_type):
        self.buffer_type_dict[vname] = buffer_type
        if "disk" == buffer_type:
            dump_dir = osp.join(self.buffer_dir, vname)
            mmcv.mkdir_or_exist(dump_dir)

    def update_buffer_types(self, buffer_type_dict):
        self.buffer_type_dict.update(buffer_type_dict)
        self.global_buffer_type = self.buffer_type_dict.get("global", "memory")

    def load_or_create_and_dump_var(self, vname, function=None, parameters=None):
        """Load from memory, if it doesn't exist, execute the function and buffer the result to memory"""
        content = self.buffer_dict.get(vname)
        if content is None and function is not None:
            if parameters is None:
                content = function()
            else:
                content = function(parameters)
            self.buffer_dict[vname] = content
        return content

    def load_or_create_and_dump(self, vname, vid, buffer_type=None,
                                function=None, parameters=None,
                                backend="torch"):
        content = self.load(vname, vid, backend=backend)
        if content is None and function is not None:
            if parameters is None:
                content = function()
            else:
                content = function(parameters)
            self.dump(vname, vid, content, buffer_type=buffer_type, backend=backend)
        return content

    def dump(self, vname, vid, vcontent, buffer_type=None, backend="torch"):
        vid = str(vid)
        if buffer_type is None:
            buffer_type = self.buffer_type_dict.get(vname, self.global_buffer_type)
        self.update_buffer_type(vname, buffer_type)
        vcontent = self.scatter_to_cpu(vcontent)
        if "memory" == buffer_type:
            buffer = self.buffer_dict.get(vname, dict())
            buffer[vid] = vcontent
            self.buffer_dict[vname] = buffer
        elif "disk" == buffer_type:
            dump_dir = osp.join(self.buffer_dir, vname)
            if "torch" == backend:
                torch.save(vcontent, osp.join(dump_dir, vid + ".pth"))
        return

    def load(self, vname, vids, backend="torch"):
        contents = []
        buffer_type = self.buffer_type_dict.get(vname, self.global_buffer_type)
        if not isinstance(vids, list):
            vids = [vids]
        vids = [str(vid) for vid in vids]
        for vid in vids:
            if "memory" == buffer_type:
                buffer = self.buffer_dict.get(vname, dict())
                content = buffer.get(vid, None)
                contents.append(content)
            elif "disk" == buffer_type:
                dump_dir = osp.join(self.buffer_dir, vname)
                if "torch" == backend:
                    content_fp = osp.join(dump_dir, vid + ".pth")
                    if osp.exists(content_fp):
                        contents.append(torch.load(content_fp))
                    else:
                        contents.append(None)
        contents = scatter(contents, [0])[0]
        if len(vids) == 1:
            return contents[0]
        return contents

    @staticmethod
    def scatter_to_cpu(inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs.detach().cpu()
        elif isinstance(inputs, (list, tuple)):
            if isinstance(inputs, tuple):
                inputs = list(inputs)
            return [Buffer.scatter_to_cpu(x) for x in inputs]
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                inputs[k] = Buffer.scatter_to_cpu(v)
        return inputs