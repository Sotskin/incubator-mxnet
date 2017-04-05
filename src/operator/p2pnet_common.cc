/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_common.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <chrono>
#include <sstream>
#include <string>
#include <thread>
#include <iterator>
#include "./p2pnet_common.h"
#include "./ctpl_stl.h"

using namespace std::chrono;

namespace mxnet {
namespace op {

P2PNet::P2PNet() {
  zmq_context_ =  zmq_ctx_new();
  zmq_ctx_set(zmq_context_, ZMQ_IO_THREADS, 10);
  server_ = zmq_socket(zmq_context_, ZMQ_ROUTER);
  internal_server_ = zmq_socket(zmq_context_, ZMQ_ROUTER);
  int value = 0;
  zmq_setsockopt(internal_server_, ZMQ_RCVHWM, &value, sizeof(value));
  value = 1024;
  zmq_setsockopt(internal_server_, ZMQ_BACKLOG, &value, sizeof(value));
  zmq_setsockopt(server_, ZMQ_BACKLOG, &value, sizeof(value));
  is_main_start_ = false;
  is_bind_ = false;
  main_thread_ = nullptr;
  recv_thread_pool_ =  new ctpl::thread_pool(8);
  internal_request_queue_size_ = 0;
  internal_request_queue_.resize(kRequestQueueSize);
}

P2PNet::~P2PNet() {
  zmq_close(internal_server_);
  zmq_close(server_);
  delete recv_thread_pool_;
}

bool P2PNet::Init(const std::string& address) {
  for (unsigned i = 0; i < internal_request_queue_size_; i++) {
    CHECK(internal_request_queue_[i]->is_fulfilled);
    delete internal_request_queue_[i];
    internal_request_queue_[i] = nullptr;
  }
  internal_request_queue_size_ = 0;

  if (!is_bind_) {
    srand(time(nullptr));
    std::ostringstream address_with_proto;
    address_with_proto << "tcp://" << address;
    int ret = zmq_bind(server_, address_with_proto.str().c_str());
    if (ret == 0) {
      zmq_bind(internal_server_, "inproc://mxnet_local_request");
      is_bind_ = true;
      std::cout << "Successfully bound to " << address_with_proto.str()
                << std::endl;
      return true;
    } else {
      std::cout << "Failed to bind to " << address_with_proto.str()
                << std::endl;
      return false;
    }
  }
  return true;
};

static int RecvWithIdentity(void* socket, std::string* identity, void* buffer,
                            int len) {
  // TODO: Use constant instead of the magic number 8.
  char identity_buffer[8];
  zmq_recv(socket, identity_buffer, 8, 0);
  *identity = std::string(identity_buffer, 8);
  zmq_recv(socket, buffer, 0, 0);
  return zmq_recv(socket, buffer, len, 0);
}

// Create a random identity with exact 8 characters.
static std::string CreateIdentity() {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<std::mt19937::result_type> dist6(0, 9999999);
  return std::to_string(dist6(rng) + 90000000);
}

void DoSendOnComplete(void* data, void* hint) {
  (void) data;
  P2PNet::Request* request = reinterpret_cast<P2PNet::Request*>(hint);
  P2PNetDebugger::Get().PrintTime(
      "DoSend of %u calls on_complete with %u bytes",
      request->tensor_id, request->buffer_size);
  request->is_fulfilled = true;
  request->on_complete();
}

void P2PNet::DoSend(struct Request* request) {
  P2PNetDebugger::Get().PrintTime("DoSend of %u", request->tensor_id);
  std::string receiver_identity = tensor_to_receiver_map_[request->tensor_id];
  tensor_to_send_request_map_.erase(request->tensor_id);
  tensor_to_receiver_map_.erase(request->tensor_id);
  zmq_msg_t msg;
  zmq_msg_init_size(&msg, receiver_identity.size());
  memcpy(zmq_msg_data(&msg), receiver_identity.c_str(), receiver_identity.size());
  zmq_msg_send(&msg, server_, ZMQ_SNDMORE);
  zmq_msg_init_size(&msg, receiver_identity.size());
  memcpy(zmq_msg_data(&msg), &request->tensor_id, sizeof(request->tensor_id));
  zmq_msg_send(&msg, server_, ZMQ_SNDMORE);
  zmq_msg_init_data(&msg, (void*)request->buffer, request->buffer_size,
                    DoSendOnComplete, request);
  zmq_msg_send(&msg, server_, 0);
}

void P2PNet::DoRequestRecv(struct Request* request) {
  void* request_socket;
  auto it = recv_request_sockets_.find(request->address);
  if (it == recv_request_sockets_.end()) {
    request_socket = zmq_socket(zmq_context_, ZMQ_DEALER);
    std::ostringstream address;
    address << "tcp://" << request->address;
    //std::string identity = CreateIdentity();
    std::string identity = std::to_string(request->tensor_id + 90000000);
    zmq_setsockopt(request_socket, ZMQ_IDENTITY, identity.c_str(),
                   identity.size());
    zmq_connect(request_socket, address.str().c_str());
    recv_request_sockets_[request->address] = request_socket;
    // TODO: We can use a more efficient way to avoid copying everytime.
    // But this may not be very critical.
    poll_items_count_++;
    zmq_pollitem_t* new_poll_items = new zmq_pollitem_t[poll_items_count_];
    std::copy(poll_items_, poll_items_ + poll_items_count_ - 1, new_poll_items);
    delete poll_items_;
    poll_items_ = new_poll_items;
    poll_items_[poll_items_count_ - 1] = {request_socket, 0, ZMQ_POLLIN, 0};
    recv_request_tensor_id_[poll_items_count_ - 1] = request->tensor_id;
    recv_poll_indices_[request_socket]  = poll_items_count_ - 1;
  } else {
    request_socket = it->second;
    // FIXME
    poll_items_[recv_poll_indices_[request_socket]].events = ZMQ_POLLIN;
  }
  // TODO: Currently, we only have one and the only one request to the remote
  // worker. Therefore, we assume that the request content is the tensor_id.
  // This may not robust when the communication becomes more complicated.
  zmq_send(request_socket, "", 0, ZMQ_SNDMORE); // ZMQ delimiter message.
  zmq_send(request_socket, &request->tensor_id, sizeof(request->tensor_id), 0);
}

void P2PNet::DoRecv(void* socket) {
  uint64_t tensor_id;
  zmq_recv(socket, &tensor_id, sizeof(tensor_id), 0);
  auto it = tensor_to_recv_request_map_.find(tensor_id);
  if (it == tensor_to_recv_request_map_.end()) {
    std::cout << "DoRecv got something unusual " << tensor_id << std::endl;
    CHECK(false);
  }
  struct Request* request = internal_request_queue_[it->second];
  tensor_to_recv_request_map_.erase(it);
  //// FIXME
  //poll_items_[i].events = ZMQ_POLLOUT;
  //recv_thread_pool_->push(DoRecvOncomplete, request,
                          //poll_items_[i].socket);
  P2PNetDebugger::Get().PrintTime("Recv of %u", request->tensor_id);
  if (P2PNetDebugger::Get().Level() &
      P2PNetDebugger::kDebugNoReceiveCopy) {
    zmq_msg_t msg;
    zmq_msg_init_size(&msg, request->buffer_size);
    zmq_msg_recv(&msg, socket, 0);
  } else {
    zmq_recv(socket, request->buffer, request->buffer_size, 0);
  }
  P2PNetDebugger::Get().PrintTime("Recv of %u calls on_complete",
                                  request->tensor_id);
  request->is_fulfilled = true;
  request->on_complete();
}

void DoRecvOncomplete(int id, P2PNet::Request* request, void* socket) {
  (void) id;
  P2PNetDebugger::Get().PrintTime("Recv of %u", request->tensor_id);
  zmq_recv(socket, request->buffer, 0, 0);
  zmq_recv(socket, request->buffer, request->buffer_size, 0);
  P2PNetDebugger::Get().PrintTime("Recv of %u calls on_complete",
                                  request->tensor_id);
  request->on_complete();
}


void P2PNet::DoInternalRequest(size_t index) {
  struct Request* request = internal_request_queue_[index];
  request->is_fulfilled = false;
  if (request->type == SendRequest) {
    P2PNetDebugger::Get().PrintTime("Received %u SendRequest",
                                    request->tensor_id);
    tensor_to_send_request_map_[request->tensor_id] = index;
    if (tensor_to_receiver_map_.find(request->tensor_id) !=
        tensor_to_receiver_map_.end()) {
      DoSend(request);
    }
  } else if (request->type == RecvRequest) {
    P2PNetDebugger::Get().PrintTime("Received %u RecvRequest",
                                    request->tensor_id);
    tensor_to_recv_request_map_[request->tensor_id] = index;
    DoRequestRecv(request);
  }
}

void P2PNet::DoExternalRequest() {
  std::string identity;
  uint64_t tensor_id = 0;
  RecvWithIdentity(server_, &identity, &tensor_id, sizeof(tensor_id));
  tensor_to_receiver_map_[tensor_id] = identity;
  auto it = tensor_to_send_request_map_.find(tensor_id);
  P2PNetDebugger::Get().PrintTime("Received %u ExternalRequest", tensor_id);
  if (it != tensor_to_send_request_map_.end()) {
    struct Request* request = internal_request_queue_[it->second];
    DoSend(request);
  }
}

void P2PNet::Main() {
  poll_items_count_ = 2;
  poll_items_ = new zmq_pollitem_t[poll_items_count_];
  poll_items_[0] = {internal_server_, 0, ZMQ_POLLIN, 0};
  poll_items_[1] = {server_, 0, ZMQ_POLLIN, 0};
  long timeout =
    (P2PNetDebugger::Get().Level() & P2PNetDebugger::kDebugPrintPending) ?
    5000 : -1;

  while (true) {
    int ret = zmq_poll(poll_items_, poll_items_count_, timeout);
    if (ret == 0) {
      CHECK(P2PNetDebugger::Get().Level() & P2PNetDebugger::kDebugPrintPending);
      for (size_t i = 0; i < internal_request_queue_size_; i++) {
        struct Request* request = internal_request_queue_[i];
        if (!request->is_fulfilled) {
          if (request->type == SendRequest) {
            std::cout << "Pending SendRequest";
          } else {
            std::cout << "Pending RecvRequest";
          }
          std::cout << " tensor_id: " << request->tensor_id;
          std::cout << " address: " << request->address << std::endl;
        }
      }
      std::cout << "==========" << std::endl;
      continue;
    }
    if (poll_items_[0].revents & ZMQ_POLLIN) { // internal request
      std::string identity;
      size_t index;
      RecvWithIdentity(internal_server_, &identity, &index, sizeof(index));
      DoInternalRequest(index);
    }
    if (poll_items_[1].revents & ZMQ_POLLIN) {
      DoExternalRequest();
    }
    for (unsigned i = 2; i < poll_items_count_; i++) {
      if (poll_items_[i].revents & ZMQ_POLLIN) {
        DoRecv(poll_items_[i].socket);
      }
    }
  }
}

void P2PNet::Start() {
  if (!is_main_start_) {
    main_thread_ = new std::thread(&P2PNet::Main, this);
    is_main_start_ = true;
  }
}

void P2PNet::DoRequest(struct Request* request) {
  void* request_socket = zmq_socket(zmq_context_, ZMQ_REQ);
  //std::string identity = CreateIdentity();
  std::string identity = std::to_string(request->tensor_id + 90000000);
  zmq_setsockopt(request_socket, ZMQ_IDENTITY, identity.c_str(), 8);
  int ret = zmq_connect(request_socket, "inproc://mxnet_local_request");
  CHECK(ret == 0);
  size_t index = internal_request_queue_size_.fetch_add(1);
  internal_request_queue_[index] = request;
  ret = zmq_send(request_socket, &index, sizeof(index), 0);
  CHECK((ret == sizeof(index))) << "Ret = " << ret << " Errno = " << errno;
  zmq_close(request_socket);
}

}  // namespace op
}  // namespace mxnet
