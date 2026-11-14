"""Checks for chat prompt handling and local checkpoint discovery."""
import importlib.util,sys,tempfile,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('saphriel_chat',ROOT/'tools/chat_server.py');chat=importlib.util.module_from_spec(spec);sys.modules[spec.name]=chat;spec.loader.exec_module(chat)
class ChatChecks(unittest.TestCase):
 def test_current_model_default_priority(self):
  self.assertEqual(chat.default_profile({'instruction-192':1,'latest-story':1}),'latest-story')
  self.assertEqual(chat.default_profile({'latest-story':1,'latest-instruction':1}),'latest-instruction')
  self.assertIsNone(chat.default_profile({}))
 def test_chat_format_and_story_mode(self):
  messages=[dict(role='user',content='Hi'),dict(role='assistant',content='Hello'),dict(role='user',content='Where?')]
  self.assertEqual(chat.prompt_text(messages,'chat'),'Instruction:\nHi\nResponse:\nHello\nInstruction:\nWhere?\nResponse:\n')
  self.assertEqual(chat.prompt_text(messages,'story'),'Where?')
 def test_invalid_requests(self):
  for messages in ([],[dict(role='system',content='x')],[dict(role='assistant',content='x')],[dict(role='user',content=' ')],[dict(role='user',content='x'*6001)]):
   with self.assertRaises(ValueError):chat.prompt_text(messages,'chat')
  for value in ({'temperature':float('nan')},{'max_tokens':True},{'max_tokens':129},{'temperature':-1}):
   with self.assertRaises(ValueError):chat.settings(value)
 def test_paths_must_remain_local(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder)
   with self.assertRaises(ValueError):chat.inside(root/'../outside.pt',root/'results')
 def test_verified_reference_discovery(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);(root/'results').mkdir();(root/'results/model.pt').write_bytes(b'fixture')
   selection=dict(reference_checkpoint='results/model.pt',reference_checkpoint_sha256='filehash',reference_parameter_sha256='paramhash',unique_parameters=5115072)
   (root/'WorkingSelection.json').write_text(json.dumps(selection));_,profiles=chat.discover(root)
   self.assertEqual(list(profiles),['approved-reference']);self.assertEqual(profiles['approved-reference'].parameters,5115072)
   self.assertNotIn('checkpoint',profiles['approved-reference'].public())
if __name__=='__main__':unittest.main()
